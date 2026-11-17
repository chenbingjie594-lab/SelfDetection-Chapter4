#!/usr/bin/env python3
"""Export aligned pre-NMS YOLO candidates through the exact validator data path."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from chapter4_detection.models.sfdfm import register_sfdfm

class RawCaptureValidator(DetectionValidator):
    records: dict = {}
    latest: dict | None = None
    output_dir: Path
    head_meta: dict = {}

    def init_metrics(self, model):
        super().init_metrics(model)
        native = model.model if getattr(model, "format", None) == "pt" else model
        head = native.model[-1]
        type(self).head_meta = {"nc":head.nc,"reg_max":head.reg_max,"strides":[float(v) for v in head.stride],"nl":head.nl}
        def hook(_module, _inputs, output):
            if not isinstance(output, tuple) or not isinstance(output[1], dict):
                raise RuntimeError("Unexpected Detect output; refusing raw-candidate export")
            decoded, raw = output
            self.latest={"decoded":decoded.detach().float().cpu(),"boxes":raw["boxes"].detach().float().cpu(),"logits":raw["scores"].detach().float().cpu(),"shapes":[list(f.shape) for f in raw["feats"]]}
        self._hook=head.register_forward_hook(hook)

    def update_metrics(self,preds,batch):
        if self.latest is None:raise RuntimeError("Detect hook did not capture raw output")
        if len(batch["im_file"])!=1:raise RuntimeError("Stage9 exporter requires batch=1")
        name=Path(batch["im_file"][0]).name; raw=self.latest
        ratio_pad=batch["ratio_pad"][0]; ori=batch["ori_shape"][0]
        stem=Path(name).stem; target=type(self).output_dir/f"{stem}.npz"
        np.savez_compressed(target,decoded=raw["decoded"][0].numpy(),raw_boxes=raw["boxes"][0].numpy(),logits=raw["logits"][0].numpy())
        type(self).records[name]={"file":target.name,"input_shape":list(batch["img"].shape[-2:]),"original_shape":[int(v) for v in ori],"ratio_pad":[[float(v) for v in ratio_pad[0]],[float(v) for v in ratio_pad[1]]],"feature_shapes":raw["shapes"]}
        super().update_metrics(preds,batch);self.latest=None

    def postprocess(self, preds):
        # AutoBackend may apply backend-specific output handling after the native
        # Detect hook. Use the exact tensor received by the validator for decoded
        # boxes/scores, while retaining hooked raw regression/logit tensors.
        if self.latest is None:raise RuntimeError("Detect hook did not run before validator postprocess")
        final = preds[0] if isinstance(preds,(tuple,list)) else preds
        self.latest["decoded"] = final.detach().float().cpu()
        return super().postprocess(preds)

def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',default='0');p.add_argument('--imgsz',type=int,default=1536);p.add_argument('--name',required=True);a=p.parse_args()
    register_sfdfm();a.output_dir.mkdir(parents=True,exist_ok=True);RawCaptureValidator.records={};RawCaptureValidator.output_dir=a.output_dir
    metrics=YOLO(str(a.weights)).val(validator=RawCaptureValidator,data=str(a.data),imgsz=a.imgsz,batch=1,device=a.device,conf=.001,iou=.7,max_det=300,plots=False)
    manifest={"model":a.name,"weights":str(a.weights.resolve()),"head":RawCaptureValidator.head_meta,"records":RawCaptureValidator.records,"formal":{"map50":float(metrics.box.map50),"map50_95":float(metrics.box.map)}}
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
