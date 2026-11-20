#!/usr/bin/env python3
"""Frozen Stage18 trainer entry point (not launched when pre-training gate fails)."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import torch,ultralytics,yaml
from ultralytics import YOLO
from chapter4_detection.models.rglcd import RGLCDTrainer
def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=(0,1,2),required=True);p.add_argument('--init',type=Path,required=True);p.add_argument('--device',default='0');p.add_argument('--project',type=Path,required=True);a=p.parse_args()
 data=Path('chapter4_detection/results/stage10/cv_protocol')/f'fold{a.fold}'/'data.yaml'
 common=yaml.safe_load(Path('chapter4_detection/configs/common.yaml').read_text());keys=('batch','workers','optimizer','lr0','lrf','momentum','weight_decay','warmup_epochs','warmup_momentum','warmup_bias_lr','box','cls','dfl','hsv_h','hsv_s','hsv_v','translate','scale','fliplr','mosaic','mixup','close_mosaic','amp','deterministic');kw={k:common[k] for k in keys};name=f'rglcd_fold{a.fold}_seed42';run=a.project/name;run.mkdir(parents=True,exist_ok=False);meta={'protocol':'stage10_matched_stage19','fold':a.fold,'seed':42,'data':str(data.resolve()),'initialization':str(a.init.resolve()),'initialization_sha256':hashlib.sha256(a.init.read_bytes()).hexdigest(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'checkpoint_selection':'last.pt_epoch150','validation_during_training':False,'common':kw};(run/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 model=YOLO(str(a.init));kw.update(data=str(data),epochs=150,imgsz=1536,seed=42,device=a.device,project=str(a.project),name=name,exist_ok=True,pretrained=True,val=False,plots=False,patience=0,save=True);model.train(trainer=RGLCDTrainer,**kw)
if __name__=='__main__':main()
