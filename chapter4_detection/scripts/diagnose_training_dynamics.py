#!/usr/bin/env python3
"""Fresh diagnostic run recording the first 300 successful optimizer updates."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import torch,yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from chapter4_detection.models.nbr_de import NBRDE,register_nbr_de

def norm(module,prefix=""):
 return sum((0 if p.grad is None else float(p.grad.detach().float().norm())**2) for n,p in module.named_parameters() if n.startswith(prefix))**.5
class DynamicsTrainer(DetectionTrainer):
 rows=[];output_csv:Path;max_updates=300;successful_updates=0;attempts=0
 def optimizer_step(self):
  module=next(m for m in self.model.modules() if isinstance(m,NBRDE));module.dynamics_enabled=True
  scale_before=self.scaler.get_scale();self.scaler.unscale_(self.optimizer)
  before=float(module.gamma.detach()); finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters())
  row={"attempt":self.attempts,"successful_update":self.successful_updates,"epoch":self.epoch,"gamma_before":before,"gamma_grad":0 if module.gamma.grad is None else float(module.gamma.grad.detach()),"query_grad":norm(module,"query"),"key_grad":norm(module,"key"),"value_grad":norm(module,"value"),"difference_grad":norm(module,"difference"),"gate_grad":norm(module,"gate"),"p3_before_grad":norm(self.model.model[16]),"p3_after_grad":norm(self.model.model[18]),"relative_output_change":module.last_dynamics.get("relative_output_change"),"injection_ratio":module.last_dynamics.get("injection_ratio"),"gate_mean":module.last_dynamics.get("gate_mean"),"loss_box":float(self.loss_items.get("box_loss",float("nan"))),"loss_cls":float(self.loss_items.get("cls_loss",float("nan"))),"loss_dfl":float(self.loss_items.get("dfl_loss",float("nan"))),"grad_finite_before_step":finite,"scaler_before":scale_before}
  torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=10.0);self.scaler.step(self.optimizer);self.scaler.update();skipped=self.scaler.get_scale()<scale_before;row.update(gamma_after=float(module.gamma.detach()),step_skipped=skipped,scaler_after=self.scaler.get_scale());self.rows.append(row);self.attempts+=1
  if not skipped:self.successful_updates+=1
  self.optimizer.zero_grad();
  if self.ema:self.ema.update(self.model)
  self.output_csv.parent.mkdir(parents=True,exist_ok=True)
  with self.output_csv.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=self.rows[0],lineterminator="\n");w.writeheader();w.writerows(self.rows)
  if self.successful_updates>=self.max_updates:self.stop=True
def main():
 p=argparse.ArgumentParser();p.add_argument("--weights",type=Path,required=True);p.add_argument("--common",type=Path,default=Path("chapter4_detection/configs/common.yaml"));p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--device",default="0");p.add_argument("--updates",type=int,default=300);a=p.parse_args();register_nbr_de();c=yaml.safe_load(a.common.read_text());a.output_dir.mkdir(parents=True,exist_ok=True);DynamicsTrainer.rows=[];DynamicsTrainer.output_csv=a.output_dir/"training_dynamics.csv";DynamicsTrainer.max_updates=a.updates;DynamicsTrainer.successful_updates=0;DynamicsTrainer.attempts=0
 model=YOLO(str(a.weights));next(m for m in model.model.modules() if isinstance(m,NBRDE)).dynamics_enabled=True
 keys=("batch","workers","optimizer","lr0","lrf","momentum","weight_decay","warmup_epochs","warmup_momentum","warmup_bias_lr","box","cls","dfl","hsv_h","hsv_s","hsv_v","translate","scale","fliplr","mosaic","mixup","close_mosaic","amp","deterministic")
 kw={k:c[k] for k in keys};kw.update(data=c["data"],epochs=150,imgsz=1536,seed=42,device=a.device,project=str(a.output_dir.resolve()),name="diagnostic_run",exist_ok=False,pretrained=True,val=False,plots=False,patience=0)
 model.train(trainer=DynamicsTrainer,**kw);summary={"diagnostic_only":True,"requested_successful_updates":a.updates,"successful_updates":sum(not r["step_skipped"] for r in DynamicsTrainer.rows),"optimizer_attempts":len(DynamicsTrainer.rows),"note":"fresh seed42 diagnostic run; not a reconstruction of historical E3"};(a.output_dir/"training_dynamics_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
