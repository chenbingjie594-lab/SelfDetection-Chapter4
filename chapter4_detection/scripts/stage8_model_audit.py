#!/usr/bin/env python3
"""Create and verify Stage8 SFDFM matched semantic initializations."""
from __future__ import annotations
import argparse,hashlib,json,random
from pathlib import Path
import numpy as np,torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from chapter4_detection.models.sfdfm import SFDFM,register_sfdfm
from chapter4_detection.scripts.stage7_initialization_audit import real_batch,flat_outputs,grad_stats

def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def mapped_key(key):
 head,layer,suffix=key.split(".",2);i=int(layer);target=i if i<=19 else i+3
 return f"model.{target}.{suffix}"
def main():
 p=argparse.ArgumentParser();p.add_argument("--baseline",type=Path,required=True);p.add_argument("--image",type=Path,required=True);p.add_argument("--label",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--imgsz",type=int,default=640);p.add_argument("--device",default="cpu");a=p.parse_args();register_sfdfm();seed_all(a.seed)
 base=YOLO(str(a.baseline)).model.to(a.device);target=YOLO("chapter4_detection/models/yolo11s_sfdfm.yaml").model.to(a.device);src=base.state_dict();dst=target.state_dict();copied=[];missing=[]
 for k,v in src.items():
  q=mapped_key(k)
  if q in dst and dst[q].shape==v.shape:dst[q].copy_(v);copied.append((k,q,v.numel()))
  else:missing.append({"source":k,"target":q,"shape":list(v.shape)})
 target.load_state_dict(dst);shared_bad=[]
 for k,q,_ in copied:
  if not torch.equal(base.state_dict()[k],target.state_dict()[q]):shared_bad.append((k,q))
 a.output.parent.mkdir(parents=True,exist_ok=True);wrapper=YOLO("yolo11s.yaml");wrapper.model=target.cpu();wrapper.save(a.output);target=target.to(a.device)
 batch=real_batch(a.image,a.label,a.imgsz,a.device)
 for m in (base,target):m.args=get_cfg();m.train();m.zero_grad(set_to_none=True)
 out0=base(batch["img"]);loss0,_=base(batch);loss0.sum().backward();out1=target(batch["img"]);loss1,_=target(batch);loss1.sum().backward()
 f0=flat_outputs(out0);f1=flat_outputs(out1);relative=float(sum((x-y).float().square().sum() for x,y in zip(f0,f1)).sqrt()/(sum(y.float().square().sum() for y in f0).sqrt()+1e-12))
 module=next(m for m in target.modules() if isinstance(m,SFDFM));new_stats=grad_stats(module);copied_targets={q for _,q,_ in copied};new_keys=[k for k in target.state_dict() if k not in copied_targets]
 report={"seed":a.seed,"baseline":str(a.baseline.resolve()),"output":str(a.output.resolve()),"baseline_sha256":hashlib.sha256(a.baseline.read_bytes()).hexdigest(),"output_sha256":hashlib.sha256(a.output.read_bytes()).hexdigest(),"copied_tensors":len(copied),"copied_elements":sum(x[2] for x in copied),"baseline_elements":sum(x.numel() for x in src.values()),"missing_source":missing,"shared_value_mismatches":shared_bad,"new_parameter_keys":new_keys,"initial_raw_output_relative_l2":relative,"initial_loss_baseline":float(loss0.detach().sum()),"initial_loss_sfdfm":float(loss1.detach().sum()),"new_branch_gradients":new_stats,"all_new_trainable_gradients_nonzero":all(v["grad_norm"] not in (None,0.0) and v["finite"] for v in new_stats.values())}
 a.output.with_suffix(".audit.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({k:report[k] for k in ("copied_tensors","copied_elements","baseline_elements","shared_value_mismatches","initial_raw_output_relative_l2","all_new_trainable_gradients_nonzero")},indent=2))
if __name__=="__main__":main()
