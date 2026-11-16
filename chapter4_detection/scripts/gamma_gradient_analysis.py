#!/usr/bin/env python3
"""Verify gamma-zero gradient routing with the real YOLO detection loss."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from chapter4_detection.models.nbr_de import NBRDE,register_nbr_de
from chapter4_detection.scripts.stage7_initialization_audit import real_batch

def group(module,prefix):
 vals=[]
 for n,p in module.named_parameters():
  if n.startswith(prefix):
   g=p.grad;vals.append(0.0 if g is None else float(g.detach().float().norm())**2)
 return sum(vals)**.5
def snapshot(model,module,precision,update,gamma_before,gamma_after,skipped):
 adjacent_before=model.model[16];adjacent_after=model.model[18]
 return {"precision":precision,"optimizer_update":update,"gamma_before":gamma_before,"gamma_after":gamma_after,"step_skipped":skipped,"gamma_grad":None if module.gamma.grad is None else float(module.gamma.grad.float().norm()),"query_grad":group(module,"query"),"key_grad":group(module,"key"),"value_grad":group(module,"value"),"difference_grad":group(module,"difference"),"gate_grad":group(module,"gate"),"p3_before_grad":group(adjacent_before,""),"p3_after_grad":group(adjacent_after,""),"finite":all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())}
def run(weights,batch,amp):
 model=YOLO(str(weights)).model.cuda().train();model.args=get_cfg();module=next(x for x in model.modules() if isinstance(x,NBRDE));opt=torch.optim.AdamW(model.parameters(),lr=1e-3);scaler=torch.amp.GradScaler("cuda",enabled=amp);rows=[]
 successful=0;attempt=0
 while successful<2 and attempt<20:
  opt.zero_grad(set_to_none=True);before=float(module.gamma.detach())
  with torch.autocast("cuda",dtype=torch.float16,enabled=amp): loss,_=model(batch)
  scale_before=scaler.get_scale();scaler.scale(loss.sum()).backward();scaler.unscale_(opt);pre=snapshot(model,module,"AMP" if amp else "FP32",successful,before,before,False);scaler.step(opt);scaler.update();after=float(module.gamma.detach());skipped=scaler.get_scale()<scale_before;pre["gamma_after"]=after;pre["step_skipped"]=skipped;pre["attempt"]=attempt;pre["scaler_before"]=scale_before;pre["scaler_after"]=scaler.get_scale();pre["loss"]=float(loss.detach().sum());rows.append(pre);successful+=int(not skipped);attempt+=1
 if successful<2: raise RuntimeError(f"Failed to obtain two successful {'AMP' if amp else 'FP32'} optimizer updates")
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument("--weights",type=Path,required=True);p.add_argument("--image",type=Path,required=True);p.add_argument("--label",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--imgsz",type=int,default=640);a=p.parse_args();register_nbr_de();batch=real_batch(a.image,a.label,a.imgsz,"cuda");rows=run(a.weights,batch,False)+run(a.weights,batch,True);a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/"gamma_gradient_analysis.json").write_text(json.dumps(rows,indent=2)+"\n")
 with (a.output_dir/"gamma_gradient_analysis.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0],lineterminator="\n");w.writeheader();w.writerows(rows)
 print(json.dumps(rows,indent=2))
if __name__=="__main__":main()
