#!/usr/bin/env python3
"""Frozen Stage18 trainer entry point (not launched when pre-training gate fails)."""
import argparse
from pathlib import Path
from ultralytics import YOLO
from chapter4_detection.models.rglcd import RGLCDTrainer
def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=(0,1,2),required=True);p.add_argument('--init',type=Path,required=True);p.add_argument('--device',default='0');p.add_argument('--project',type=Path,required=True);a=p.parse_args()
 data=Path('chapter4_detection/results/stage10/cv_protocol')/f'fold{a.fold}'/'data.yaml'
 model=YOLO(str(a.init));model.train(trainer=RGLCDTrainer,data=str(data),epochs=150,imgsz=1536,batch=1,patience=0,optimizer='auto',lr0=.01,seed=42,device=a.device,workers=4,project=str(a.project),name=f'rglcd_fold{a.fold}_seed42',exist_ok=False,val=False,save=True)
if __name__=='__main__':main()
