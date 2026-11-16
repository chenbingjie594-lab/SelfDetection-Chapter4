#!/usr/bin/env python3
"""Train-only multi-scale separability diagnosis for matched YOLO11s checkpoints."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import cv2,numpy as np,torch
from ultralytics import YOLO

LAYERS={4:"backbone_p2",6:"backbone_p3",16:"neck_p3",19:"neck_p4",22:"neck_p5"}
def prep(im,size):
 h,w=im.shape[:2];g=min(size/w,size/h);nw,nh=round(w*g),round(h*g);r=cv2.resize(im,(nw,nh));out=np.full((size,size,3),114,np.uint8);px=(size-nw)//2;py=(size-nh)//2;out[py:py+nh,px:px+nw]=r
 return torch.from_numpy(out[:,:,::-1].copy()).permute(2,0,1).unsqueeze(0).float().div(255),(w,h,g,px,py),out
def region_stats(f,box):
 _,c,h,w=f.shape;x1,y1,x2,y2=box;ix1=max(0,int(np.floor(x1*w)));ix2=min(w,max(ix1+1,int(np.ceil(x2*w))));iy1=max(0,int(np.floor(y1*h)));iy2=min(h,max(iy1+1,int(np.ceil(y2*h))))
 pad=max(1,int(np.ceil(max(ix2-ix1,iy2-iy1))*.75));rx1=max(0,ix1-pad);rx2=min(w,ix2+pad);ry1=max(0,iy1-pad);ry2=min(h,iy2+pad)
 roi=f[0,:,iy1:iy2,ix1:ix2].float();ring=f[0,:,ry1:ry2,rx1:rx2].float();mask=torch.ones(ring.shape[-2:],dtype=torch.bool,device=f.device);mask[iy1-ry1:iy2-ry1,ix1-rx1:ix2-rx1]=False;bg=ring[:,mask]
 rv=roi.flatten(1).mean(1);bv=bg.mean(1);response=float(roi.abs().mean());bgresp=float(bg.abs().mean());dist=float(torch.linalg.vector_norm(rv-bv)/np.sqrt(c));contrast=(response-bgresp)/(bgresp+1e-12)
 energy=roi.abs().mean(0).flatten();p=energy/(energy.sum()+1e-12);entropy=float(-(p[p>0]*torch.log2(p[p>0])).sum()/np.log2(max(2,p.numel())))
 return response,bgresp,contrast,dist,entropy
def bootstrap(rows,n,seed):
 rng=np.random.default_rng(seed);out={}
 for cls in ("all",0,1):
  sub=[r for r in rows if cls=="all" or r["class_id"]==cls];groups=defaultdict(list)
  for r in sub:groups[r["source_group"]].append(r)
  names=list(groups);out[str(cls)]={}
  for layer in LAYERS.values():
   z=[r for r in sub if r["layer"]==layer];point=float(np.mean([r["feature_contrast"] for r in z]));vals=[]
   for _ in range(n):
    pick=rng.choice(names,len(names),replace=True);q=[r["feature_contrast"] for g in pick for r in groups[g] if r["layer"]==layer];vals.append(np.mean(q))
   out[str(cls)][layer]={"contrast_mean":point,"ci95":[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],"distance_mean":float(np.mean([r["feature_distance"] for r in z]))}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--runs",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--imgsz",type=int,default=1536);p.add_argument("--device",default="cuda:0");p.add_argument("--bootstrap",type=int,default=1000);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 recs=[r for r in json.loads(a.manifest.read_text())["records"] if r["split"]=="train"];rows=[];heat=[]
 for seed in (42,3407,2026):
  model=YOLO(str(a.runs/f"matched_e0_seed{seed}/weights/best.pt")).model.to(a.device).eval();cap={};handles=[model.model[i].register_forward_hook(lambda m,x,y,n=n:cap.__setitem__(n,y.detach())) for i,n in LAYERS.items()]
  chosen=set()
  for rec in recs:
   im=cv2.imread(rec["image"]);x,geom,canvas=prep(im,a.imgsz);x=x.to(a.device)
   with torch.inference_mode():model(x)
   W,H,g,px,py=geom
   for j,line in enumerate(Path(rec["label"]).read_text().splitlines()):
    z=line.split();cls=int(z[0]);cx,cy,bw,bh=map(float,z[1:]);x1=((cx-bw/2)*W*g+px)/a.imgsz;x2=((cx+bw/2)*W*g+px)/a.imgsz;y1=((cy-bh/2)*H*g+py)/a.imgsz;y2=((cy+bh/2)*H*g+py)/a.imgsz
    for name,f in cap.items():
     response,bg,contrast,dist,entropy=region_stats(f,(x1,y1,x2,y2));rows.append({"seed":seed,"image":rec["image_name"],"canonical_image_id":rec["canonical_image_id"],"source_group":rec["source_group"],"class_id":cls,"roi_index":j,"layer":name,"feature_response":response,"background_response":bg,"feature_contrast":contrast,"feature_distance":dist,"local_entropy":entropy})
    if seed==42 and cls not in chosen:
     chosen.add(cls);heat.append((cls,rec["image_name"],canvas.copy(),{k:v.abs().mean(1)[0].cpu().numpy() for k,v in cap.items()},(x1,y1,x2,y2)))
  for h in handles:h.remove()
 with (a.output_dir/"feature_statistics.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0],lineterminator="\n");w.writeheader();w.writerows(rows)
 summary={"protocol":"train_only_three_matched_e0_checkpoints","rows":len(rows),"bootstrap":bootstrap(rows,a.bootstrap,42)};(a.output_dir/"feature_diagnosis.json").write_text(json.dumps(summary,indent=2)+"\n")
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(len(heat),5,figsize=(16,3.3*len(heat)));axs=np.atleast_2d(axs)
 for row,(cls,name,canvas,maps,box) in enumerate(heat):
  for col,(layer,hm) in enumerate(maps.items()):
   axs[row,col].imshow(hm,cmap="inferno");axs[row,col].set_title(f"{'Flash' if cls==0 else 'Black'} {layer}");axs[row,col].axis("off")
 fig.tight_layout();fig.savefig(a.output_dir/"typical_feature_heatmaps.png",dpi=180);plt.close(fig)
 print(json.dumps({"rows":len(rows),"images":len(recs)},indent=2))
if __name__=="__main__":main()
