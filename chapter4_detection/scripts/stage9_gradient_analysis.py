#!/usr/bin/env python3
"""Train-only cls/reg gradient cosine analysis at shared Neck P3."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import cv2,numpy as np,torch
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.cfg import DEFAULT_CFG_DICT
from chapter4_detection.models.sfdfm import register_sfdfm

def prep(im,lines,size):
 h,w=im.shape[:2];g=min(size/w,size/h);nw,nh=round(w*g),round(h*g);r=cv2.resize(im,(nw,nh));out=np.full((size,size,3),114,np.uint8);px=(size-nw)//2;py=(size-nh)//2;out[py:py+nh,px:px+nw]=r
 cls=[];boxes=[]
 for line in lines:
  z=line.split();c=int(z[0]);x,y,bw,bh=map(float,z[1:]);cls.append(c);boxes.append([(x*w*g+px)/size,(y*h*g+py)/size,bw*w*g/size,bh*h*g/size])
 x=torch.from_numpy(out[:,:,::-1].copy()).permute(2,0,1).unsqueeze(0).float().div(255)
 return x,torch.tensor(cls,dtype=torch.float32).view(-1,1),torch.tensor(boxes,dtype=torch.float32).view(-1,4)

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoints',nargs='+',required=True);p.add_argument('--names',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='0');p.add_argument('--imgsz',type=int,default=1536);a=p.parse_args()
 if len(a.checkpoints)!=len(a.names):raise ValueError('checkpoint/name count differs')
 register_sfdfm();records=[r for r in json.loads(a.manifest.read_text())['records'] if r['split']=='train'];rows=[];device=torch.device(f'cuda:{a.device}' if str(a.device).isdigit() else a.device)
 for ckpt,name in zip(a.checkpoints,a.names):
  model=YOLO(ckpt).model.to(device).train();model.requires_grad_(True)
  if isinstance(model.args,dict):model.args=IterableSimpleNamespace(**{**DEFAULT_CFG_DICT,**model.args})
  criterion=v8DetectionLoss(model);captured={}
  def pre(_m,inputs):captured['p3']=inputs[0][0];captured['p3'].retain_grad()
  handle=model.model[-1].register_forward_pre_hook(pre)
  for rec in records:
   lines=Path(rec['label']).read_text().splitlines();im=cv2.imread(rec['image']);x,cls,boxes=prep(im,lines,a.imgsz);x=x.to(device);cls=cls.to(device);boxes=boxes.to(device)
   batch={'img':x,'batch_idx':torch.zeros(len(cls),device=device),'cls':cls,'bboxes':boxes}
   preds=model(x);loss,_=criterion.loss(preds,batch);f=captured['p3']
   gc=torch.autograd.grad(loss[1],f,retain_graph=True)[0].float();gr=torch.autograd.grad(loss[0]+loss[2],f)[0].float()
   nc=float(gc.norm());nr=float(gr.norm());den=nc*nr;cos=float((gc*gr).sum()/den) if den>1e-20 else None
   classes=sorted({int(v) for v in cls.cpu().flatten()});areas=[float(v[2]*v[3]) for v in boxes.cpu()]
   rows.append({'checkpoint':name,'image':rec['image_name'],'source_group':rec['source_group'],'classes':','.join(map(str,classes)),'instances':len(lines),'min_normalized_area':min(areas),'cls_grad_norm':nc,'reg_grad_norm':nr,'cosine':cos,'near_zero':den<=1e-20,'negative':cos is not None and cos<0})
   model.zero_grad(set_to_none=True)
  handle.remove();del model;torch.cuda.empty_cache()
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 summary={}
 for name in a.names:
  z=[r for r in rows if r['checkpoint']==name];valid=[r['cosine'] for r in z if r['cosine'] is not None]
  summary[name]={'images':len(z),'cosine_mean':float(np.mean(valid)),'cosine_median':float(np.median(valid)),'negative_fraction':float(np.mean([v<0 for v in valid])),'near_zero':sum(r['near_zero'] for r in z),'cls_norm_mean':float(np.mean([r['cls_grad_norm'] for r in z])),'reg_norm_mean':float(np.mean([r['reg_grad_norm'] for r in z]))}
 a.output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
