#!/usr/bin/env python3
"""Fixed Stage10-matched PCRF grouped-CV trainer."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import torch,ultralytics,yaml
from ultralytics import YOLO
from chapter4_detection.models.pcrf import PCRFDetect,PCRFTrainer
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(3),required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--initialized',type=Path,required=True);p.add_argument('--project',type=Path,required=True);p.add_argument('--device',default='0');a=p.parse_args();a.project=a.project.resolve();a.data=a.data.resolve();a.initialized=a.initialized.resolve();m=YOLO(str(a.initialized));assert isinstance(m.model.model[-1],PCRFDetect);name=f'pcrf_fold{a.fold}_seed42';run=a.project/name;run.mkdir(parents=True,exist_ok=False);common=yaml.safe_load(Path('chapter4_detection/configs/common.yaml').read_text());meta={'protocol':'stage11_pcrf_group_cv_v1','fold':a.fold,'seed':42,'epochs':150,'validation_during_training':False,'checkpoint_selection':'last.pt_epoch150','initialization_sha256':sha(a.initialized),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'torch':torch.__version__,'ultralytics':ultralytics.__version__};(run/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n');keys=('batch','workers','optimizer','lr0','lrf','momentum','weight_decay','warmup_epochs','warmup_momentum','warmup_bias_lr','box','cls','dfl','hsv_h','hsv_s','hsv_v','translate','scale','fliplr','mosaic','mixup','close_mosaic','amp','deterministic');kw={k:common[k] for k in keys};kw.update(data=str(a.data),epochs=150,imgsz=1536,seed=42,device=a.device,project=str(a.project),name=name,exist_ok=True,pretrained=True,val=False,plots=False,patience=0,save=True);m.train(trainer=PCRFTrainer,**kw);last=run/'weights/last.pt';meta.update(completed=True,checkpoint_sha256=sha(last));(run/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
