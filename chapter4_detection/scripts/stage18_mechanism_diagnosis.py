#!/usr/bin/env python3
"""Train-only RGLCD feature statistics; never reads holdout labels."""
import argparse,csv,json,random
from pathlib import Path
import numpy as np,torch
from ultralytics import YOLO
from chapter4_detection.models.rglcd import RGLCD
from chapter4_detection.scripts.stage7_initialization_audit import real_batch
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--train-list',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--imgsz',type=int,default=640);a=ap.parse_args();random.seed(42)
 model=YOLO(str(a.weights)).model.cpu().eval();mod=next(x for x in model.modules() if isinstance(x,RGLCD));cap={};hook=model.model[-1].register_forward_pre_hook(lambda m,x:cap.update(p3=x[0][0].detach()))
 rows=[]
 for ip in a.train_list.read_text().splitlines():
  image=Path(ip);label=Path(str(image).replace('/images/','/labels/')).with_suffix('.txt');batch=real_batch(image,label,a.imgsz,'cpu')
  with torch.no_grad():model(batch['img']);p3=cap['p3'];mean,var,d,r,count=mod.statistics(p3);res=r*mod.phi(d)
  h,w=p3.shape[-2:]
  boxes=batch['bboxes'];classes=batch['cls'].flatten().long()
  occupied=[]
  for j,b in enumerate(boxes):
   cx,cy,bw,bh=[float(x) for x in b];x=min(w-1,max(0,int(cx*w)));y=min(h-1,max(0,int(cy*h)));occupied.append((cx,cy,bw,bh))
   vals={'image':image.name,'kind':'GT','class':int(classes[j]),'tiny':min(bw*a.imgsz,bh*a.imgsz)<8,'d_abs':float(d[0,:,y,x].abs().mean()),'d_signed':float(d[0,:,y,x].mean()),'variance':float(var[0,:,y,x].mean()),'reliability':float(r[0,:,y,x].mean()),'residual_abs':float(res[0,:,y,x].abs().mean()),'ring_may_include_gt':bool(bw*w>1 or bh*h>1)};rows.append(vals)
  # Four deterministic grid backgrounds, excluding every known GT box.
  for gy,gx in ((.2,.2),(.2,.8),(.8,.2),(.8,.8)):
   if any(abs(gx-cx)<=bw/2+.02 and abs(gy-cy)<=bh/2+.02 for cx,cy,bw,bh in occupied):continue
   x=int(gx*(w-1));y=int(gy*(h-1));rows.append({'image':image.name,'kind':'BACKGROUND','class':-1,'tiny':False,'d_abs':float(d[0,:,y,x].abs().mean()),'d_signed':float(d[0,:,y,x].mean()),'variance':float(var[0,:,y,x].mean()),'reliability':float(r[0,:,y,x].mean()),'residual_abs':float(res[0,:,y,x].abs().mean()),'ring_may_include_gt':False})
 hook.remove();a.output.mkdir(parents=True,exist_ok=True)
 with (a.output/'rglcd_train_feature_samples.csv').open('w',newline='') as f:wri=csv.DictWriter(f,fieldnames=list(rows[0]));wri.writeheader();wri.writerows(rows)
 def stats(sel):
  z=[q for q in rows if sel(q)];return {'n':len(z),**{k:float(np.mean([q[k] for q in z])) for k in ('d_abs','d_signed','variance','reliability','residual_abs')}}
 summary={'resolution':a.imgsz,'scope':'fold0 train only','gt':stats(lambda q:q['kind']=='GT'),'background':stats(lambda q:q['kind']=='BACKGROUND'),'flash':stats(lambda q:q['class']==0),'black':stats(lambda q:q['class']==1),'tiny':stats(lambda q:q['kind']=='GT' and q['tiny']),'ring_overlap_proxy_rate':float(np.mean([q['ring_may_include_gt'] for q in rows if q['kind']=='GT'])),'reliability_saturation_gt_below_0.1':float(np.mean([q['reliability']<.1 for q in rows if q['kind']=='GT'])),'reliability_saturation_gt_above_0.99':float(np.mean([q['reliability']>.99 for q in rows if q['kind']=='GT']))}
 (a.output/'rglcd_mechanism_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
