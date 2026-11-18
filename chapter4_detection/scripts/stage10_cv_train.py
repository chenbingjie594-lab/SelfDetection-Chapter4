#!/usr/bin/env python3
"""Fixed 150-epoch, no-validation, last.pt-only Stage10 CV trainer."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
import torch,ultralytics,yaml
from ultralytics import YOLO
from chapter4_detection.models.cbcd import CBCDDetect,CBCDTrainer
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['e0','cbcd'],required=True);p.add_argument('--fold',type=int,choices=range(3),required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='0');p.add_argument('--project',type=Path,required=True);p.add_argument('--e0-init',type=Path,required=True);p.add_argument('--cbcd-init',type=Path,required=True);a=p.parse_args()
 init=a.e0_init if a.model=='e0' else a.cbcd_init;model=YOLO(str(init));
 if a.model=='cbcd' and not isinstance(model.model.model[-1],CBCDDetect):raise RuntimeError('CBCD initialization lost CBCDDetect')
 name=f'{a.model}_fold{a.fold}_seed{a.seed}';run=a.project/name;run.mkdir(parents=True,exist_ok=False)
 common=yaml.safe_load(Path('chapter4_detection/configs/common.yaml').read_text());meta={'protocol':'stage10_group_cv_v1','model':a.model,'fold':a.fold,'seed':a.seed,'epochs':150,'validation_during_training':False,'checkpoint_selection':'last.pt_epoch150','data':str(a.data.resolve()),'initialization':str(init.resolve()),'initialization_sha256':sha(init),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'torch':torch.__version__,'ultralytics':ultralytics.__version__}
 (run/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 keys=('batch','workers','optimizer','lr0','lrf','momentum','weight_decay','warmup_epochs','warmup_momentum','warmup_bias_lr','box','cls','dfl','hsv_h','hsv_s','hsv_v','translate','scale','fliplr','mosaic','mixup','close_mosaic','amp','deterministic')
 kw={k:common[k] for k in keys};kw.update(data=str(a.data),epochs=150,imgsz=1536,seed=a.seed,device=a.device,project=str(a.project),name=name,exist_ok=True,pretrained=True,val=False,plots=False,patience=0,save=True)
 model.train(trainer=CBCDTrainer if a.model=='cbcd' else None,**kw)
 last=run/'weights/last.pt';assert last.exists();meta.update(checkpoint_sha256=sha(last),completed=True);(run/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
