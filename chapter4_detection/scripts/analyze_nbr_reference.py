#!/usr/bin/env python3
"""Diagnose sparse-attention and local NBR references without retraining."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from ultralytics import YOLO

from chapter4_detection.models.nbr_de import NBRDE, register_nbr_de
from chapter4_detection.scripts.evaluate_original import load_gt
from chapter4_detection.scripts.freeze_diagnostic_thresholds import local_contrast


def stats(x: list[float]) -> dict:
    a = np.asarray(x, dtype=float)
    return {"count": len(a), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "median": float(np.median(a)), "q25": float(np.quantile(a, .25)), "q75": float(np.quantile(a, .75))}


def attention_details(module: NBRDE, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    q, k, v = module.query(x), module.key(x), module.value(x)
    pad = max(max(abs(y), abs(z)) for y, z in module.offsets)
    valid = x.new_ones((x.shape[0], 1, *x.shape[-2:]))
    logits, values = [], []
    for dy, dx in module.offsets:
        ks = module._sample(k, dy, dx, pad); values.append(module._sample(v, dy, dx, pad))
        mask = module._sample(valid, dy, dx, pad).bool()
        score = (q * ks).sum(1, keepdim=True) * q.shape[1] ** -0.5
        logits.append(score.masked_fill(~mask, torch.finfo(score.dtype).min))
    weights = torch.softmax(torch.cat(logits, 1).float(), 1).to(x.dtype)
    background = sum(weights[:, i:i+1] * value for i, value in enumerate(values))
    variance = sum(weights[:, i:i+1] * (value - background).square().mean(1, keepdim=True)
                   for i, value in enumerate(values))
    energy = sum(weights[:, i:i+1] * value.square().mean(1, keepdim=True)
                 for i, value in enumerate(values))
    return background, module.local(x), weights, variance / (energy + 1e-6)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True); p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True); p.add_argument("--thresholds", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--device", default="0"); p.add_argument("--imgsz", type=int, default=1536)
    a = p.parse_args(); register_nbr_de()
    thresholds = json.loads(a.thresholds.read_text())["classes"]["1"]["contrast_tertiles"]
    model = YOLO(str(a.weights)); paths = sorted(x for x in a.images.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png"})
    first = cv2.imread(str(paths[0])); model.predict(first, imgsz=a.imgsz, conf=.001, device=a.device, verbose=False)
    module = next(m for m in model.predictor.model.model.modules() if isinstance(m, NBRDE)); module.export_features = True
    keys = ("entropy", "max_weight", "neighbor_inconsistency", "cosine_F_Batt", "residual_att_abs",
            "residual_local_abs", "reference_difference_abs")
    values = {region: {key: [] for key in keys} for region in ("defect", "background", "low_contrast_black")}
    rows=[]
    for path in paths:
        image=cv2.imread(str(path)); gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY); h,w=image.shape[:2]
        model.predict(image, imgsz=a.imgsz, conf=.001, device=a.device, verbose=False)
        x=module.last_features["F"]
        with torch.no_grad(): batt, local, weights, inconsistency=attention_details(module,x)
        _,_,fh,fw=x.shape; defect=torch.zeros((fh,fw),dtype=torch.bool,device=x.device); low_black=torch.zeros_like(defect)
        classes,boxes=load_gt(a.labels/f"{path.stem}.txt",w,h)
        for cls,box in zip(classes.tolist(),boxes.numpy()):
            x1,y1,x2,y2=box; ix1,ix2=max(0,int(x1*fw/w)),min(fw,int(np.ceil(x2*fw/w))); iy1,iy2=max(0,int(y1*fh/h)),min(fh,int(np.ceil(y2*fh/h)))
            defect[iy1:iy2,ix1:ix2]=True
            if int(cls)==1 and local_contrast(gray,box)<=thresholds[0]: low_black[iy1:iy2,ix1:ix2]=True
        entropy=-(weights.float().clamp_min(1e-12)*weights.float().clamp_min(1e-12).log()).sum(1)[0]
        metrics={"entropy":entropy,"max_weight":weights.max(1).values[0].float(),"neighbor_inconsistency":inconsistency[0,0].float(),
                 "cosine_F_Batt":F.cosine_similarity(x.float(),batt.float(),dim=1)[0],
                 "residual_att_abs":(x-batt).float().abs().mean(1)[0],"residual_local_abs":(x-local).float().abs().mean(1)[0],
                 "reference_difference_abs":(batt-local).float().abs().mean(1)[0]}
        for region,mask in (("defect",defect),("background",~defect),("low_contrast_black",low_black)):
            if not mask.any(): continue
            row={"image":path.name,"region":region,"cells":int(mask.sum())}
            for key,tensor in metrics.items():
                array=tensor[mask].detach().cpu().numpy()
                if region=="background" and len(array)>512: array=array[np.linspace(0,len(array)-1,512,dtype=int)]
                values[region][key].extend(array.tolist()); row[key]=float(array.mean())
            rows.append(row)
    center=module.local.weight.detach()[:,:,1,1].abs().sum(); total=module.local.weight.detach().abs().sum()
    output={"weights":str(a.weights),"samples":len(paths),"offsets":[list(x) for x in module.offsets],
            "regions":{region:{key:stats(x) for key,x in metrics.items()} for region,metrics in values.items()},
            "local_reference_center_abs_weight_fraction":float(center/total),
            "local_reference_note":"3x3 depthwise convolution includes its center coefficient; this quantifies potential self-information mixing.",
            "interpretation_guard":"Residual magnitude alone does not establish reference quality; validation GT is used only for post-hoc grouping."}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(output,indent=2)+"\n")
    with a.csv.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(output,indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
