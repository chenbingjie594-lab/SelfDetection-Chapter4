#!/usr/bin/env python3
"""Warm, synchronized B0/B1/B2 inference benchmark on a fixed image list."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import cv2,numpy as np,torch
from ultralytics import YOLO
def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--image-list',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='3');p.add_argument('--images',type=int,default=20);p.add_argument('--repeats',type=int,default=3);a=p.parse_args();paths=[Path(x) for x in a.image_list.read_text().splitlines() if x.strip()][:a.images];raw=[cv2.imread(str(x)) for x in paths];model=YOLO(str(a.weights));cuda=torch.cuda.is_available()
 out={'protocol':{'warmup':3,'images':len(paths),'repeats':a.repeats,'cuda_synchronized':True,'includes_external_B2_resize':True},'conditions':{}}
 for c in ['B0','B1','B2']:
  def prep(im):return im if c!='B2' else cv2.resize(cv2.resize(im,(1536,384),interpolation=cv2.INTER_AREA),(4096,1024),interpolation=cv2.INTER_LINEAR)
  imgsz=1536 if c=='B0' else (1024,4096);rect=c!='B0';warm=prep(raw[0])
  for _ in range(3):model.predict(warm,imgsz=imgsz,rect=rect,conf=.001,iou=.7,max_det=300,device=a.device,verbose=False)
  if cuda:torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
  vals=[];pre=[]
  for _ in range(a.repeats):
   for im in raw:
    t=time.perf_counter();x=prep(im);pre.append(time.perf_counter()-t);torch.cuda.synchronize();t=time.perf_counter();model.predict(x,imgsz=imgsz,rect=rect,conf=.001,iou=.7,max_det=300,device=a.device,verbose=False);torch.cuda.synchronize();vals.append(time.perf_counter()-t)
  out['conditions'][c]={'preprocess_external_seconds_mean':float(np.mean(pre)),'model_pipeline_seconds_mean':float(np.mean(vals)),'total_seconds_mean':float(np.mean(pre)+np.mean(vals)),'total_seconds_sd':float(np.std(np.asarray(pre)+np.asarray(vals))),'peak_gpu_memory_bytes':int(torch.cuda.max_memory_allocated()),'actual_shape':[1,3,1536,1536] if c=='B0' else [1,3,1024,4096]}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
