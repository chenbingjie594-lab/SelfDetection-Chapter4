#!/usr/bin/env python3
"""Initialization and real YOLO-loss gradient audit for RGLCD."""
import argparse,copy,json
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from chapter4_detection.models.rglcd import RGLCD,RGLCDDetect,replace_detect_with_rglcd
from chapter4_detection.scripts.stage7_initialization_audit import real_batch
def norm(p):return None if p.grad is None else float(p.grad.detach().float().norm())
def module_grad(m):
 vals=[float(p.grad.detach().float().norm())**2 for p in m.parameters() if p.grad is not None];return sum(vals)**.5
def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--image',type=Path,required=True);p.add_argument('--label',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--imgsz',type=int,default=640);a=p.parse_args()
 wrap=YOLO(str(a.baseline));model=wrap.model.cpu().eval();old=copy.deepcopy(model.model[-1]).eval();replace_detect_with_rglcd(model);head=model.model[-1];batch=real_batch(a.image,a.label,a.imgsz,'cpu');cap={}
 h=head.register_forward_pre_hook(lambda m,x:cap.update(feats=[q.detach() for q in x[0]]))
 with torch.no_grad():model(batch['img']);f=cap['feats'];o=old.forward_head(f,**old.one2many);n=head.forward_head(f,**head.one2many)
 h.remove();init={'box_max_abs':float((o['boxes']-n['boxes']).abs().max()),'score_max_abs':float((o['scores']-n['scores']).abs().max()),'decoded_max_abs':float((old(f)[0]-head(f)[0]).abs().max()) if not model.training else None}
 model.args=get_cfg();model.train();opt=torch.optim.AdamW(model.parameters(),lr=1e-3);rows=[];module=next(x for x in model.modules() if isinstance(x,RGLCD))
 for step in range(3):
  opt.zero_grad(set_to_none=True);before=float(module.gamma.detach());loss,_=model(batch);loss.sum().backward();row={'step':step,'loss':float(loss.sum()),'gamma_before':before,'gamma_grad':norm(module.gamma),'phi_grad':norm(module.phi.weight),'p3_cls_grad':module_grad(head.cv3[0]),'p3_reg_grad':module_grad(head.cv2[0]),'finite':all(q.grad is None or torch.isfinite(q.grad).all() for q in model.parameters()),**module.last_stats};opt.step();row['gamma_after']=float(module.gamma.detach());rows.append(row)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'initial':init,'steps':rows,'module_type':type(head).__name__},indent=2)+'\n');print(json.dumps({'initial':init,'steps':rows},indent=2))
if __name__=='__main__':main()
