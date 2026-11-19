#!/usr/bin/env python3
"""Matched Fold0 training with assignment tracing at preregistered epochs."""
from __future__ import annotations
import argparse,csv,hashlib,json,subprocess
from pathlib import Path
import torch,ultralytics,yaml
from ultralytics import YOLO
from chapter4_detection.models import assignment_diagnostic as diag

EPOCHS={1,3,10,25,50,100,150}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_records(path):
 if not diag.RECORDER.records:return
 keys=sorted({k for r in diag.RECORDER.records for k in r})
 with path.open('w',newline='') as f:w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(diag.RECORDER.records)
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--init',type=Path,required=True);p.add_argument('--project',type=Path,required=True);p.add_argument('--device',default='1');a=p.parse_args();name='e0_fold0_seed42_assignment_diagnostic';run=a.project/name;run.mkdir(parents=True,exist_ok=False);trace=run/'assignment_dynamics.csv'
 meta={'protocol':'stage13_assignment_diagnostic_v1','baseline_protocol':'stage10_group_cv_v1','fold':0,'seed':42,'epochs':150,'trace_epochs':sorted(EPOCHS),'data':str(a.data.resolve()),'initialization':str(a.init.resolve()),'initialization_sha256':sha(a.init),'git_parent':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'validation_during_training':False,'checkpoint_selection':'last.pt_epoch150'};(run/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 diag.RECORDER.records=[];diag.RECORDER.enabled=False
 model=YOLO(str(a.init))
 def epoch_start(trainer):
  e=trainer.epoch+1;diag.RECORDER.epoch=e;diag.RECORDER.enabled=e in EPOCHS
 def epoch_end(trainer):
  if trainer.epoch+1 in EPOCHS:write_records(trace)
 model.add_callback('on_train_epoch_start',epoch_start);model.add_callback('on_train_epoch_end',epoch_end)
 common=yaml.safe_load(Path('chapter4_detection/configs/common.yaml').read_text());keys=('batch','workers','optimizer','lr0','lrf','momentum','weight_decay','warmup_epochs','warmup_momentum','warmup_bias_lr','box','cls','dfl','hsv_h','hsv_s','hsv_v','translate','scale','fliplr','mosaic','mixup','close_mosaic','amp','deterministic');kw={k:common[k] for k in keys};kw.update(data=str(a.data),epochs=150,imgsz=1536,seed=42,device=a.device,project=str(a.project),name=name,exist_ok=True,pretrained=True,val=False,plots=False,patience=0,save=True)
 model.train(trainer=diag.AssignmentDiagnosticTrainer,**kw);write_records(trace);last=run/'weights/last.pt';meta.update(completed=last.exists(),checkpoint_sha256=sha(last) if last.exists() else None,trace_rows=len(diag.RECORDER.records));(run/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
