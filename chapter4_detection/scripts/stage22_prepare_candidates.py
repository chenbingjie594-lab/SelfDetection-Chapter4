#!/usr/bin/env python3
"""Freeze E0 post-NMS candidates for CIR; GT is used only in training mode."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from ultralytics import YOLO

def iou(a,b):
 lt=np.maximum(a[:2],b[:2]);rb=np.minimum(a[2:],b[2:]);q=np.maximum(rb-lt,0).prod();aa=np.maximum(a[2:]-a[:2],0).prod();bb=np.maximum(b[2:]-b[:2],0).prod();return q/(aa+bb-q+1e-12)
def labels(image):
 p=Path(str(image).replace('/images/','/labels/')).with_suffix('.txt');out=[]
 for j,z in enumerate(p.read_text().splitlines()):
  if not z.strip():continue
  c,x,y,w,h=map(float,z.split());out.append({'gt_id':j,'class_id':int(c),'xyxy':[(x-w/2)*4096,(y-h/2)*1024,(x+w/2)*4096,(y+h/2)*1024]})
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--images-list',type=Path,required=True);p.add_argument('--weights',type=Path);p.add_argument('--frozen-predictions',type=Path);p.add_argument('--mode',choices=('train','eval'),required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();images=[Path(x) for x in a.images_list.read_text().splitlines() if x.strip()]
 if a.frozen_predictions:
  frozen=json.loads(a.frozen_predictions.read_text())['images'];pred={im.name:frozen[im.name]['predictions'] for im in images}
 else:
  assert a.weights;model=YOLO(str(a.weights));results=model.predict(source=[str(x) for x in images],imgsz=1536,conf=.001,iou=.7,max_det=300,device='cpu',verbose=False,rect=False)
  pred={}
  for im,r in zip(images,results):pred[im.name]=[{'class_id':int(c),'confidence':float(s),'xyxy':[float(v) for v in b]} for b,s,c in zip(r.boxes.xyxy.cpu(),r.boxes.conf.cpu(),r.boxes.cls.cpu())]
 out=[];stats={'positive':0,'negative':0,'ignored':0,'Flash':0,'Black':0,'tiny':0,'iou_0.30_0.50':0,'iou_0.50_0.75':0,'iou_0.75_1.00':0}
 for im in images:
  pp=pred[im.name];gg=labels(im) if a.mode=='train' else [];pairs=[]
  for pi,q in enumerate(pp):
   for gi,g in enumerate(gg):
    if q['class_id']==g['class_id']:pairs.append((iou(np.array(q['xyxy']),np.array(g['xyxy'])),pi,gi))
  usedp=set();usedg=set();matched={}
  for ov,pi,gi in sorted(pairs,reverse=True):
   if ov<.30:break
   if pi in usedp or gi in usedg:continue
   usedp.add(pi);usedg.add(gi);matched[pi]=(gg[gi],ov)
  for pi,q in enumerate(pp):
   rec={**q,'image':im.name,'image_path':str(im.resolve()),'fold':a.fold,'candidate_id':f'{a.fold}:{im.name}:{pi}'}
   if a.mode=='eval':rec['positive']=False;out.append(rec);continue
   if pi in matched:
    g,ov=matched[pi];b=np.array(q['xyxy']);t=np.array(g['xyxy']);bw=b[2]-b[0];bh=b[3]-b[1];rec.update(positive=True,gt_id=g['gt_id'],gt_class_id=g['class_id'],gt_xyxy=g['xyxy'],match_iou=ov,target=[((t[0]+t[2])-(b[0]+b[2]))/(2*bw),((t[1]+t[3])-(b[1]+b[3]))/(2*bh),float(np.log((t[2]-t[0])/bw)),float(np.log((t[3]-t[1])/bh))]);stats['positive']+=1;stats['Flash' if g['class_id']==0 else 'Black']+=1;stats['tiny']+=min((t[2]-t[0])*.375,(t[3]-t[1])*.375)<8;stats['iou_0.30_0.50' if ov<.5 else 'iou_0.50_0.75' if ov<.75 else 'iou_0.75_1.00']+=1;out.append(rec)
   else:
    maxany=max((iou(np.array(q['xyxy']),np.array(g['xyxy'])) for g in gg),default=0)
    if maxany>.10:stats['ignored']+=1;continue
    rec.update(positive=False,target=[0,0,0,0],match_iou=0);stats['negative']+=1;out.append(rec)
 payload={'protocol':'conf=.001,class-aware NMS iou=.7,max_det=300; train match class-aware greedy IoU>=.30; uncertain unmatched overlap>.10 ignored','fold':a.fold,'mode':a.mode,'weights':str(a.weights.resolve()) if a.weights else None,'weights_sha256':hashlib.sha256(a.weights.read_bytes()).hexdigest() if a.weights else None,'images':len(images),'stats':{k:int(v) for k,v in stats.items()},'candidates':out};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,default=lambda x:x.item() if hasattr(x,'item') else x)+'\n');print(json.dumps({**payload['stats'],'saved':len(out)},indent=2))
if __name__=='__main__':main()
