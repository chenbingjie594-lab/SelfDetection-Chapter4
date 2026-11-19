#!/usr/bin/env python3
"""Export B1/B2 native-shape final predictions and thresholded pre-NMS candidates."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import cv2,numpy as np,torch
from ultralytics import YOLO

def sync(cuda):
 if cuda:torch.cuda.synchronize()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--image-list',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--condition',choices=['B1','B2'],required=True);p.add_argument('--device',default='3');p.add_argument('--warmup',type=int,default=3);a=p.parse_args()
 paths=[Path(x) for x in a.image_list.read_text().splitlines() if x.strip()];a.output.parent.mkdir(parents=True,exist_ok=True);a.cache_dir.mkdir(parents=True,exist_ok=True);model=YOLO(str(a.weights));head=model.model.model[-1];capture={}
 def hook(_m,_i,o):
  if not(isinstance(o,tuple) and isinstance(o[1],dict)):raise RuntimeError('unexpected Detect output')
  capture['decoded']=o[0].detach().float().cpu();capture['shapes']=[list(f.shape) for f in o[1]['feats']]
 h=head.register_forward_hook(hook);cuda=str(a.device).lower()!='cpu' and torch.cuda.is_available()
 if cuda:torch.cuda.reset_peak_memory_stats()
 def prepare(im):
  if a.condition=='B1':return im
  return cv2.resize(cv2.resize(im,(1536,384),interpolation=cv2.INTER_AREA),(4096,1024),interpolation=cv2.INTER_LINEAR)
 warm=prepare(cv2.imread(str(paths[0])))
 for _ in range(a.warmup):model.predict(warm,imgsz=(1024,4096),rect=True,conf=.001,iou=.7,max_det=300,device=a.device,verbose=False)
 records={};times=[];pre=[];post=[]
 for ip in paths:
  original=cv2.imread(str(ip));im=prepare(original);capture.clear();sync(cuda);t0=time.perf_counter();
  result=model.predict(im,imgsz=(1024,4096),rect=True,conf=.001,iou=.7,max_det=300,device=a.device,verbose=False)[0];sync(cuda);total=time.perf_counter()-t0
  if 'decoded' not in capture:raise RuntimeError('Detect hook did not fire')
  if tuple(result.orig_shape)!=(1024,4096):raise RuntimeError(f'wrong original shape {result.orig_shape}')
  decoded=capture['decoded'][0];boxes=decoded[:4].T.numpy();scores=decoded[4:].T.numpy();cls=scores.argmax(1);conf=scores[np.arange(len(scores)),cls];keep=np.flatnonzero(conf>=.001)
  boxes_xyxy=np.column_stack((boxes[:,0]-boxes[:,2]/2,boxes[:,1]-boxes[:,3]/2,boxes[:,0]+boxes[:,2]/2,boxes[:,1]+boxes[:,3]/2))
  level=np.concatenate([np.full(s[2]*s[3],i+3,np.int8) for i,s in enumerate(capture['shapes'])])
  cache=a.cache_dir/f'{ip.stem}.npz';np.savez_compressed(cache,boxes_xyxy=boxes_xyxy[keep].astype(np.float32),scores=scores[keep].astype(np.float32),class_id=cls[keep].astype(np.int8),confidence=conf[keep].astype(np.float32),level=level[keep])
  preds=[] if result.boxes is None else [{'class_id':int(c),'confidence':float(s),'xyxy':b.tolist()} for b,s,c in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.boxes.cls.cpu().numpy())]
  speed={k:float(v)/1000 for k,v in result.speed.items()};pre.append(speed.get('preprocess',0));post.append(speed.get('postprocess',0));times.append(total)
  records[ip.name]={'width':4096,'height':1024,'actual_input_shape':[1,3,1024,4096],'ratio_pad':[[1.0,1.0],[0.0,0.0]],'candidate_cache':cache.name,'candidate_count':len(keep),'feature_shapes':capture['shapes'],'seconds':total,'speed_seconds':speed,'predictions':preds}
 h.remove();payload={'schema_version':1,'condition':a.condition,'protocol':'same checkpoint; native tensor 1x3x1024x4096; conf=.001; class-aware NMS IoU=.7; max_det=300','weights':str(a.weights.resolve()),'weights_sha256':sha(a.weights),'images_count':len(records),'mean_seconds_per_original':float(np.mean(times)),'mean_preprocess_seconds':float(np.mean(pre)),'mean_postprocess_seconds':float(np.mean(post)),'peak_gpu_memory_bytes':int(torch.cuda.max_memory_allocated()) if cuda else None,'images':records};a.output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({k:v for k,v in payload.items() if k!='images'},indent=2))
if __name__=='__main__':main()
