#!/usr/bin/env python3
"""Run end-to-end YOLO+MBRD inference and retain pre/post-NMS predictions."""
import argparse,json,time
from pathlib import Path
import cv2,torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from chapter4_detection.models.mbrd import replace_detect_with_mbrd
def tensor(path):
 im=cv2.imread(str(path));z=LetterBox(new_shape=(1536,1536),auto=False,stride=32)(image=im);return torch.from_numpy(z[:,:,::-1].copy()).permute(2,0,1).float()[None]/255
def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--prototypes',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--images-list',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();pd=torch.load(a.prototypes,map_location='cpu',weights_only=False);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);yolo=YOLO(str(a.weights));base=yolo.model.cpu().eval();first=Path(a.images_list.read_text().splitlines()[0]);x=tensor(first)
 with torch.no_grad():before=base(x);br=before[1] if isinstance(before,tuple) else before
 head=replace_detect_with_mbrd(base,pd['prototypes'],ck['mode']);head.mbrd.load_state_dict(ck['state_dict']);base.eval()
 with torch.no_grad():after=base(x);ar=after[1] if isinstance(after,tuple) else after
 n3=ar['feats'][0].shape[2]*ar['feats'][0].shape[3];audit={'p3_box_logits_max_abs':float((br['boxes'][:,:,:n3]-ar['boxes'][:,:,:n3]).abs().max()),'p4p5_box_logits_max_abs':float((br['boxes'][:,:,n3:]-ar['boxes'][:,:,n3:]).abs().max()),'p3_cls_logits_mean_abs':float((br['scores'][:,:,:n3]-ar['scores'][:,:,:n3]).abs().mean()),'p4p5_cls_logits_max_abs':float((br['scores'][:,:,n3:]-ar['scores'][:,:,n3:]).abs().max()),'gamma':float(head.mbrd.gamma),'regression_tower_frozen':True,'p4p5_frozen':True}
 images={};pre={};elapsed=0.;captured=[]
 def hook(_m,_i,o):captured.append(o)
 handle=head.register_forward_hook(hook)
 for path in map(Path,a.images_list.read_text().splitlines()):
  captured.clear();t=time.time();res=yolo.predict(str(path),imgsz=1536,device='cpu',conf=.001,iou=.7,max_det=300,verbose=False)[0];elapsed+=time.time()-t;h,w=res.orig_shape;images[path.name]={'width':w,'height':h,'predictions':[{'class_id':int(c),'confidence':float(s),'xyxy':b.tolist()} for b,s,c in zip(res.boxes.xyxy.cpu(),res.boxes.conf.cpu(),res.boxes.cls.cpu())]}
  out=captured[-1];yy=out[0] if isinstance(out,tuple) else out;raw=out[1] if isinstance(out,tuple) else None;yy=yy[0].cpu();box=yy[:4].T;scores=yy[4:].T;ih,iw=raw['feats'][0].shape[2]*8,raw['feats'][0].shape[3]*8;gain=min(iw/w,ih/h);px=(iw-w*gain)/2;py=(ih-h*gain)/2;anchors,classes=torch.where(scores>=.001);order=scores[anchors,classes].argsort(descending=True)[:3000];anchors=anchors[order];classes=classes[order];bounds=[];s=0
  for f in raw['feats']:s+=f.shape[2]*f.shape[3];bounds.append(s)
  rows=[]
  for j,cl in zip(anchors,classes):
   # Predictor config sets Detect.xyxy=True; the hooked decoded tensor is input-space xyxy.
   x1,y1,x2,y2=box[j];xy=[float((x1-px)/gain),float((y1-py)/gain),float((x2-px)/gain),float((y2-py)/gain)];level=0 if j<bounds[0] else (1 if j<bounds[1] else 2);rows.append({'anchor_id':int(j),'level':level,'class_id':int(cl),'confidence':float(scores[j,cl]),'xyxy':[max(0,min(w,xy[0])),max(0,min(h,xy[1])),max(0,min(w,xy[2])),max(0,min(h,xy[3]))]})
  pre[path.name]={'width':w,'height':h,'predictions':rows};print(path.name,len(rows),len(res.boxes),flush=True)
 handle.remove();out={'schema_version':1,'model':f"MBRD-{ck['mode']}",'images_count':len(images),'images':images,'elapsed_seconds':elapsed,'mean_seconds_per_original':elapsed/len(images),'audit':audit};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out)+'\n');a.output.with_name(a.output.stem+'_prenms.json').write_text(json.dumps({'images':pre})+'\n');a.output.with_name(a.output.stem+'_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
