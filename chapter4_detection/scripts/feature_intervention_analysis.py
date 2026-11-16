#!/usr/bin/env python3
"""Layer-wise Full vs bypass analysis for frozen E3 checkpoints."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import cv2,numpy as np,torch
from ultralytics import YOLO
from chapter4_detection.models.nbr_de import NBRDE,register_nbr_de

def prep(path,size):
 im=cv2.imread(str(path));h,w=im.shape[:2];g=min(size/w,size/h);nw,nh=round(w*g),round(h*g);r=cv2.resize(im,(nw,nh));out=np.full((size,size,3),114,np.uint8);px=(size-nw)//2;py=(size-nh)//2;out[py:py+nh,px:px+nw]=r
 return torch.from_numpy(out[:,:,::-1].copy()).permute(2,0,1).unsqueeze(0).float().div(255), (w,h,g,px,py)
def rel(a,b):return float((a.float()-b.float()).norm()/(b.float().norm()+1e-12))
def main():
 p=argparse.ArgumentParser();p.add_argument("--weights-dir",type=Path,required=True);p.add_argument("--images",type=Path,required=True);p.add_argument("--labels",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--imgsz",type=int,default=1536);p.add_argument("--device",default="cuda:0");a=p.parse_args();register_nbr_de();a.output_dir.mkdir(parents=True,exist_ok=True);rows=[]
 for seed in (42,3407,2026):
  model=YOLO(str(a.weights_dir/f"e3_seed{seed}/weights/best.pt")).model.to(a.device).eval();module=next(x for x in model.modules() if isinstance(x,NBRDE));captures={}
  handles=[]
  for idx,name in ((17,"P3"),(20,"P4"),(23,"P5")):
   handles.append(model.model[idx].register_forward_hook(lambda m,i,o,n=name:captures.__setitem__(n,o.detach())))
  for ip in sorted(a.images.iterdir()):
   if ip.suffix.lower() not in {".jpg",".jpeg",".png"}:continue
   x,geom=prep(ip,a.imgsz);x=x.to(a.device);runs={}
   with torch.inference_mode():
    for mode in ("full","gate_bypass","residual_bypass"):
     module.inference_intervention=mode;captures.clear();decoded,raw=model(x);runs[mode]={"P3":captures["P3"].clone(),"P4":captures["P4"].clone(),"P5":captures["P5"].clone(),"boxes":raw["boxes"].clone(),"scores":raw["scores"].clone(),"decoded":decoded.clone()}
   w,h,g,px,py=geom;fh,fw=runs["full"]["P3"].shape[-2:];mask=torch.zeros((fh,fw),dtype=torch.bool,device=a.device)
   for line in (a.labels/f"{ip.stem}.txt").read_text().splitlines():
    z=line.split();cx,cy,bw,bh=map(float,z[1:]);x1=(cx-bw/2)*w*g+px;x2=(cx+bw/2)*w*g+px;y1=(cy-bh/2)*h*g+py;y2=(cy+bh/2)*h*g+py
    ix1=max(0,int(x1/a.imgsz*fw));ix2=min(fw,max(ix1+1,int(np.ceil(x2/a.imgsz*fw))));iy1=max(0,int(y1/a.imgsz*fh));iy2=min(fh,max(iy1+1,int(np.ceil(y2/a.imgsz*fh))));mask[iy1:iy2,ix1:ix2]=True
   for mode in ("gate_bypass","residual_bypass"):
    f,b=runs["full"],runs[mode];diff=(f["P3"]-b["P3"]).float();inside=diff[...,mask];outside=diff[...,~mask]
    rows.append({"seed":seed,"image":ip.name,"bypass":mode,"p3_relative_l2":rel(f["P3"],b["P3"]),"p4_relative_l2":rel(f["P4"],b["P4"]),"p5_relative_l2":rel(f["P5"],b["P5"]),"box_raw_relative_l2":rel(f["boxes"],b["boxes"]),"class_logit_relative_l2":rel(f["scores"],b["scores"]),"decoded_candidate_relative_l2":rel(f["decoded"],b["decoded"]),"defect_region_mean_abs_change":float(inside.abs().mean()) if inside.numel() else None,"background_mean_abs_change":float(outside.abs().mean()),"defect_to_background_change_ratio":float(inside.abs().mean()/(outside.abs().mean()+1e-12)) if inside.numel() else None})
  for hnd in handles:hnd.remove()
 with (a.output_dir/"feature_intervention_per_image.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0],lineterminator="\n");w.writeheader();w.writerows(rows)
 summary={}
 for mode in ("gate_bypass","residual_bypass"):
  z=[r for r in rows if r["bypass"]==mode];summary[mode]={k:{"mean":float(np.mean([q[k] for q in z])),"sd":float(np.std([q[k] for q in z],ddof=1)),"max":float(np.max([q[k] for q in z]))} for k in rows[0] if k not in {"seed","image","bypass"}}
 (a.output_dir/"feature_intervention_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
