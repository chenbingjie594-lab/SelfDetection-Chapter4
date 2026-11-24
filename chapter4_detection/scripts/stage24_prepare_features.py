#!/usr/bin/env python3
"""Cache frozen-E0 P3 patches, exact TAL targets, and conservative background candidates."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

def label_path(im):return Path(str(im).replace('/images/','/labels/')).with_suffix('.txt')
def read_labels(im):
 p=label_path(im);z=[]
 if p.exists():
  for s in p.read_text().splitlines():
   q=list(map(float,s.split()));z.append(q)
 return z
def assigned(crit,raw,batch):
 pd,ps=raw['boxes'].permute(0,2,1).contiguous(),raw['scores'].permute(0,2,1).contiguous();ap,st=make_anchors(raw['feats'],crit.stride,.5);dtype=ps.dtype;bs=1;imgsz=torch.tensor(raw['feats'][0].shape[2:],dtype=dtype)*crit.stride[0];targets=torch.cat((batch['batch_idx'].view(-1,1),batch['cls'].view(-1,1),batch['bboxes']),1);targets=crit.preprocess(targets,bs,imgsz[[1,0,1,0]]);gl,gb=targets.split((1,4),2);mask=gb.sum(2,keepdim=True).gt_(0);pb=crit.bbox_decode(ap,pd);_,_,ts,fg,_=crit.assigner(ps.detach().sigmoid(),(pb.detach()*st).type(gb.dtype),ap*st,gl,gb,mask);return ts,fg,gb
def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--images-list',type=Path,required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--bg-per-stratum',type=int,default=32);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(a.seed+a.fold);model=YOLO(str(a.weights)).model.cpu().eval();crit=v8DetectionLoss(model);lb=LetterBox(new_shape=(1536,1536),auto=False,scale_fill=False,scaleup=True,stride=32);images=[Path(x) for x in a.images_list.read_text().splitlines()];patches=[];targets=[];base_logits=[];positive=[];meta=[];bg_vectors=[];bg_meta=[];start=time.time()
 with torch.no_grad():
  for ii,path in enumerate(images):
   im=cv2.imread(str(path));h0,w0=im.shape[:2];arr=lb(image=im);x=torch.from_numpy(arr[:,:,::-1].copy()).permute(2,0,1).float()[None]/255;out=model(x);raw=out[1] if isinstance(out,tuple) else out;f=raw['feats'][0].float();_,c,h,w=f.shape;n=h*w;labs=read_labels(path);cls=torch.tensor([[q[0]] for q in labs],dtype=torch.float32);boxes=[]
   gain=min(1536/w0,1536/h0);nw,nh=round(w0*gain),round(h0*gain);px=(1536-nw)/2;py=(1536-nh)/2
   for q in labs:
    _,cx,cy,bw,bh=q;boxes.append([(cx*w0*gain+px)/1536,(cy*h0*gain+py)/1536,bw*w0*gain/1536,bh*h0*gain/1536])
   batch={'batch_idx':torch.zeros(len(labs)),'cls':cls,'bboxes':torch.tensor(boxes,dtype=torch.float32).reshape(-1,4)};ts,fg,gb=assigned(crit,raw,batch);t3=ts[0,:n];log3=raw['scores'][0,:,:n].T;pos=t3.sum(1)>0;score=log3.sigmoid().max(1).values;feat=f[0].permute(1,2,0);norm=feat.norm(dim=2);yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij');valid=score.reshape(h,w)<.05
   # Exclude twice-expanded GT protection rectangles in input coordinates.
   valid&=(xx>=px/8)&(xx<(px+nw)/8)&(yy>=py/8)&(yy<(py+nh)/8)
   for q in boxes:
    cx,cy,bw,bh=q;x1=(cx-bw)*1536/8;y1=(cy-bh)*1536/8;x2=(cx+bw)*1536/8;y2=(cy+bh)*1536/8;valid&=~((xx>=x1)&(xx<=x2)&(yy>=y1)&(yy<=y2))
   valid&=~pos.reshape(h,w);idx=torch.where(valid.flatten())[0];nv=norm.flatten()[idx];qs=torch.quantile(nv,torch.tensor([0.,.25,.5,.75,1.])) if len(nv) else torch.zeros(5);chosen=[]
   for s in range(4):
    z=idx[(nv>=qs[s])&(nv<=(qs[s+1] if s==3 else qs[s+1]))].numpy();take=min(a.bg_per_stratum,len(z));chosen.extend(rng.choice(z,take,replace=False).tolist() if take else [])
   pad=F.pad(f,(3,3,3,3),mode='replicate')[0];sample=torch.where(pos)[0].tolist()+chosen
   for j in sample:
    y=j//w;xj=j%w;patches.append(pad[:,y:y+7,xj:xj+7].half());targets.append(t3[j].half());base_logits.append(log3[j].half());ispos=bool(pos[j]);positive.append(ispos);meta.append({'image':path.name,'image_index':ii,'p3_index':j,'x':xj,'y':y,'positive':ispos,'target_class':int(t3[j].argmax()) if ispos else -1,'texture_stratum':next((s for s in range(4) if float(norm[y,xj])<=float(qs[s+1])),3)})
   for j in chosen:
    y=j//w;xj=j%w;bg_vectors.append(feat[y,xj].half());bg_meta.append({'image':path.name,'p3_index':j,'x':xj,'y':y,'texture_norm':float(norm[y,xj]),'max_e0_score':float(score[j])})
   print(a.fold,ii+1,len(images),path.name,'pos',int(pos.sum()),'bg',len(chosen),flush=True)
 cache={'fold':a.fold,'weights':str(a.weights.resolve()),'weights_sha256':hashlib.sha256(a.weights.read_bytes()).hexdigest(),'images':[str(x) for x in images],'patches':torch.stack(patches),'targets':torch.stack(targets),'base_logits':torch.stack(base_logits),'positive':torch.tensor(positive),'meta':meta,'background_vectors':torch.stack(bg_vectors),'background_meta':bg_meta,'p3_shape':[c,h,w],'protocol':{'imgsz':1536,'patch':7,'bg_score_max':.05,'gt_protection_expand':2.,'strata':4,'bg_per_stratum':a.bg_per_stratum,'tal':'frozen E0 exact v8DetectionLoss assigner'}};torch.save(cache,a.output);rep={'fold':a.fold,'images':len(images),'samples':len(patches),'positive':sum(positive),'background':len(patches)-sum(positive),'prototype_candidates':len(bg_vectors),'p3_shape':[c,h,w],'elapsed_seconds':time.time()-start,'cache':str(a.output.resolve()),'cache_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'cache_bytes':a.output.stat().st_size};a.output.with_suffix('.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
