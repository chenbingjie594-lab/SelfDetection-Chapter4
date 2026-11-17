#!/usr/bin/env python3
"""Build matched CBCD initialization and verify classification-only semantics."""
from __future__ import annotations
import argparse,copy,hashlib,json,random
from pathlib import Path
import numpy as np,torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from chapter4_detection.models.cbcd import CBCD,replace_detect_with_cbcd
from chapter4_detection.scripts.stage7_initialization_audit import real_batch,grad_stats

def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--image',type=Path,required=True);p.add_argument('--label',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cpu');p.add_argument('--imgsz',type=int,default=640);a=p.parse_args()
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);device=f'cuda:{a.device}' if str(a.device).isdigit() else a.device
 wrapper=YOLO(str(a.baseline));model=wrapper.model.to(device).eval();old_head=copy.deepcopy(model.model[-1]).eval();transfer=replace_detect_with_cbcd(model)
 batch=real_batch(a.image,a.label,a.imgsz,device);capture={}
 def pre(_m,inputs):capture.__setitem__('features',[v.detach() for v in inputs[0]])
 handle=model.model[-1].register_forward_pre_hook(pre)
 with torch.no_grad():model(batch['img']);features=capture['features'];r0=old_head.forward_head(features,**old_head.one2many);r1=model.model[-1].forward_head(features,**model.model[-1].one2many)
 box_max_abs=float((r0['boxes']-r1['boxes']).abs().max());box_equal=box_max_abs==0;cls_rel=float((r0['scores']-r1['scores']).norm()/(r0['scores'].norm()+1e-12));handle.remove()
 model.args=get_cfg();model.train();model.requires_grad_(True);model.zero_grad(set_to_none=True);loss,_=model(batch);loss.sum().backward();module=next(m for m in model.modules() if isinstance(m,CBCD));grads=grad_stats(module)
 wrapper.model=model.cpu();a.output.parent.mkdir(parents=True,exist_ok=True);wrapper.save(a.output)
 report={'seed':a.seed,'baseline':str(a.baseline.resolve()),'output':str(a.output.resolve()),'transfer':transfer,'raw_regression_equal_within_fp_tolerance':box_equal,'raw_regression_max_abs':box_max_abs,'classification_relative_l2':cls_rel,'new_gradients':grads,'all_new_gradients_nonzero':all(v['grad_norm'] not in (None,0) and v['finite'] for v in grads.values()),'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()}
 a.output.with_suffix('.audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('raw_regression_equal_within_fp_tolerance','raw_regression_max_abs','classification_relative_l2','all_new_gradients_nonzero','sha256')},indent=2))
if __name__=='__main__':main()
