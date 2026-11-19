#!/usr/bin/env python3
"""Matched Fold0 training with per-positive TAL/DFL geometry tracing."""
from __future__ import annotations
import argparse,csv,hashlib,json,subprocess
from pathlib import Path
import torch,ultralytics,yaml
from ultralytics import YOLO
from chapter4_detection.models.assignment_geometry_monitor import RECORDER
from chapter4_detection.models.geometry_diagnostic import GeometryDiagnosticTrainer
EPOCHS={1,3,10,25,50,100,150}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p):
 if not RECORDER.records:return
 keys=sorted({k for r in RECORDER.records for k in r});
 with p.open('w',newline='') as f:w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(RECORDER.records)
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--init',type=Path,required=True);p.add_argument('--project',type=Path,required=True);p.add_argument('--device',default='2');a=p.parse_args();name='e0_fold0_seed42_geometry_diagnostic';run=a.project/name;run.mkdir(parents=True,exist_ok=False);trace=run/'positive_geometry.csv';meta={'protocol':'stage14_geometry_diagnostic_v2','baseline_protocol':'stage10_group_cv_v1','fold':0,'seed':42,'epochs':150,'trace_epochs':sorted(EPOCHS),'data':str(a.data.resolve()),'initialization':str(a.init.resolve()),'initialization_sha256':sha(a.init),'git_parent':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'validation_during_training':False};(run/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n');RECORDER.records=[];RECORDER.enabled=False;RECORDER.assignment_call=0
 model=YOLO(str(a.init));model.add_callback('on_train_epoch_start',lambda t:(setattr(RECORDER,'epoch',t.epoch+1),setattr(RECORDER,'enabled',t.epoch+1 in EPOCHS)));model.add_callback('on_train_epoch_end',lambda t:dump(trace) if t.epoch+1 in EPOCHS else None)
 c=yaml.safe_load(Path('chapter4_detection/configs/common.yaml').read_text());keys=('batch','workers','optimizer','lr0','lrf','momentum','weight_decay','warmup_epochs','warmup_momentum','warmup_bias_lr','box','cls','dfl','hsv_h','hsv_s','hsv_v','translate','scale','fliplr','mosaic','mixup','close_mosaic','amp','deterministic');kw={k:c[k] for k in keys};kw.update(data=str(a.data),epochs=150,imgsz=1536,seed=42,device=a.device,project=str(a.project),name=name,exist_ok=True,pretrained=True,val=False,plots=False,patience=0,save=True);model.train(trainer=GeometryDiagnosticTrainer,**kw);dump(trace);last=run/'weights/last.pt';meta.update(completed=last.exists(),checkpoint_sha256=sha(last) if last.exists() else None,trace_rows=len(RECORDER.records));(run/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
