#!/usr/bin/env python3
"""Replay TAL on each fold's training images without augmentation using frozen E0 checkpoints."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.loss import v8DetectionLoss
from types import SimpleNamespace
from chapter4_detection.models.assignment_monitor import AssignmentRecorder,MonitoredAssigner

class AssignmentValidator(DetectionValidator):
 recorder=AssignmentRecorder();latest=None
 def init_metrics(self,model):
  super().init_metrics(model);native=model.model if getattr(model,'format',None)=='pt' else model;self.criterion=v8DetectionLoss(native);h=dict(self.criterion.hyp) if isinstance(self.criterion.hyp,dict) else vars(self.criterion.hyp);h.setdefault('box',7.5);h.setdefault('cls',0.5);h.setdefault('dfl',1.5);self.criterion.hyp=SimpleNamespace(**h);self.criterion.assigner=MonitoredAssigner(self.criterion.assigner,type(self).recorder)
  def hook(_m,_i,o):
   if isinstance(o,tuple) and isinstance(o[1],dict):self.latest=o[1]
  self._hook=native.model[-1].register_forward_hook(hook)
 def update_metrics(self,preds,batch):
  if self.latest is None:raise RuntimeError('raw head output not captured')
  before=len(type(self).recorder.records);self.criterion.get_assigned_targets_and_loss(self.latest,batch);name=Path(batch['im_file'][0]).name
  for r in type(self).recorder.records[before:]:r['image']=name
  self.latest=None;super().update_metrics(preds,batch)

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',default='1');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);allrows=[]
 for fold in range(3):
  proto=a.repo/f'chapter4_detection/results/stage10/cv_protocol/fold{fold}';data=a.output_dir/f'fold{fold}_train_static.yaml';data.write_text(f'path: {proto}\ntrain: {proto/"train.txt"}\nval: {proto/"train.txt"}\nnames:\n  0: Flash point\n  1: Big black spots\n')
  weights=a.repo/f'chapter4_detection/results/stage10/runs/e0_fold{fold}_seed42/weights/last.pt';AssignmentValidator.recorder=AssignmentRecorder();YOLO(str(weights)).val(validator=AssignmentValidator,data=str(data),imgsz=1536,batch=1,device=a.device,conf=.001,iou=.7,max_det=300,plots=False,verbose=False)
  for r in AssignmentValidator.recorder.records:r['fold']=fold
  allrows.extend(AssignmentValidator.recorder.records)
 keys=sorted({k for r in allrows for k in r});
 with (a.output_dir/'static_assignment_instances.csv').open('w',newline='') as f:w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(allrows)
 import numpy as np
 summary={}
 for fold in range(3):
  q=[r for r in allrows if r['fold']==fold];summary[str(fold)]={'instances':len(q),'zero_positive':sum(r['positive_count']==0 for r in q),'zero_effective_weight':sum(r['target_score_sum']<=1e-9 for r in q),'positive_count_mean':float(np.mean([r['positive_count'] for r in q])),'positive_iou_mean':float(np.mean([r['positive_iou_mean'] for r in q if r['positive_iou_mean'] is not None])),'target_score_sum_mean':float(np.mean([r['target_score_sum'] for r in q]))}
 (a.output_dir/'static_assignment_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
