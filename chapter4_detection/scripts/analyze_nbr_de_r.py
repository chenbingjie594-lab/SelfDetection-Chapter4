#!/usr/bin/env python3
"""Export E4 reliability statistics and aligned mechanism visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

from chapter4_detection.models.nbr_de_r import NBRDER, register_nbr_de_r
from chapter4_detection.scripts.evaluate_original import load_gt
from chapter4_detection.scripts.freeze_diagnostic_thresholds import local_contrast


def summary(values: list[float]) -> dict:
    x=np.asarray(values,dtype=float)
    return {"count":len(x),"mean":float(x.mean()),"sd":float(x.std(ddof=1)) if len(x)>1 else 0.0,
            "min":float(x.min()),"q25":float(np.quantile(x,.25)),"median":float(np.median(x)),
            "q75":float(np.quantile(x,.75)),"max":float(x.max())}


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--images",type=Path,required=True); p.add_argument("--labels",type=Path,required=True)
    p.add_argument("--thresholds",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--visualizations",type=Path,required=True); p.add_argument("--names",nargs="*",default=[])
    p.add_argument("--device",default="0"); p.add_argument("--imgsz",type=int,default=1536); a=p.parse_args()
    register_nbr_de_r(); model=YOLO(str(a.weights)); paths=sorted(x for x in a.images.iterdir() if x.suffix.lower() in {".jpg",".jpeg",".png"})
    first=cv2.imread(str(paths[0])); model.predict(first,imgsz=a.imgsz,conf=.001,device=a.device,verbose=False)
    module=next(m for m in model.predictor.model.model.modules() if isinstance(m,NBRDER)); module.export_features=True
    low_cut=json.loads(a.thresholds.read_text())["classes"]["1"]["contrast_tertiles"][0]
    keys=("U","r","reference_difference_abs","residual_att_abs","residual_local_abs","difference_abs","gate","contribution_abs")
    values={region:{key:[] for key in keys} for region in ("defect","background","low_contrast_black")}
    a.visualizations.mkdir(parents=True,exist_ok=True)
    for path in paths:
        image=cv2.imread(str(path)); gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); h,w=image.shape[:2]
        model.predict(image,imgsz=a.imgsz,conf=.001,device=a.device,verbose=False); f=module.last_features
        _,_,fh,fw=f["F"].shape; defect=torch.zeros((fh,fw),dtype=torch.bool,device=f["F"].device); low=torch.zeros_like(defect)
        classes,boxes=load_gt(a.labels/f"{path.stem}.txt",w,h)
        for cls,box in zip(classes.tolist(),boxes.numpy()):
            x1,y1,x2,y2=box; ix1,ix2=max(0,int(x1*fw/w)),min(fw,int(np.ceil(x2*fw/w))); iy1,iy2=max(0,int(y1*fh/h)),min(fh,int(np.ceil(y2*fh/h)))
            defect[iy1:iy2,ix1:ix2]=True
            if int(cls)==1 and local_contrast(gray,box)<=low_cut: low[iy1:iy2,ix1:ix2]=True
        maps={"U":f["U"][0,0].float(),"r":f["r"][0,0].float(),
              "reference_difference_abs":(f["B_att"]-f["B_local"])[0].float().abs().mean(0),
              "residual_att_abs":(f["F"]-f["B_att"])[0].float().abs().mean(0),
              "residual_local_abs":(f["F"]-f["B_local"])[0].float().abs().mean(0),
              "difference_abs":f["D"][0].float().abs().mean(0),"gate":f["G"][0].float().mean(0),
              "contribution_abs":f["contribution"][0].float().abs().mean(0)}
        for region,mask in (("defect",defect),("background",~defect),("low_contrast_black",low)):
            if not mask.any(): continue
            for key,tensor in maps.items():
                x=tensor[mask].cpu().numpy()
                if region=="background" and len(x)>512: x=x[np.linspace(0,len(x)-1,512,dtype=int)]
                values[region][key].extend(x.tolist())
        if path.name in a.names:
            shown=cv2.resize(image,(2048,512),interpolation=cv2.INTER_AREA)
            fig,axes=plt.subplots(2,4,figsize=(18,7),constrained_layout=True); axes=axes.ravel()
            axes[0].imshow(cv2.cvtColor(shown,cv2.COLOR_BGR2RGB));axes[0].set_title("Original")
            for ax,(key,tensor) in zip(axes[1:],list(maps.items())[:7]):
                heat=cv2.resize(tensor.cpu().numpy(),(2048,512),interpolation=cv2.INTER_LINEAR)
                ax.imshow(cv2.cvtColor(shown,cv2.COLOR_BGR2RGB),alpha=.3);ax.imshow(heat,cmap="turbo",alpha=.7);ax.set_title(key)
            for ax in axes: ax.axis("off")
            fig.savefig(a.visualizations/f"{path.stem}_e4_mechanism.png",dpi=120);plt.close(fig)
    region_stats={region:{key:summary(x) for key,x in metrics.items()} for region,metrics in values.items()}
    all_r=np.asarray(values["background"]["r"]+values["defect"]["r"])
    output={"weights":str(a.weights),"images":len(paths),"gamma":float(module.gamma.detach().cpu()),
            "reliability_bias":float(module.reliability_bias.detach().cpu()),
            "reliability_slope_softplus":float(torch.nn.functional.softplus(module.reliability_slope_raw).detach().cpu()),
            "regions":region_stats,"selection_behavior":{"fraction_r_lt_0.1":float((all_r<.1).mean()),
            "fraction_r_gt_0.9":float((all_r>.9).mean()),"adaptive_not_constant":bool(all_r.std()>1e-4)},
            "visualizations":sorted(x.name for x in a.visualizations.glob("*.png"))}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2)+"\n");print(json.dumps(output,indent=2));return 0


if __name__=="__main__": raise SystemExit(main())
