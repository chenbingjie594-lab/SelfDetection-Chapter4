import torch
from ultralytics import YOLO
from chapter4_detection.models.cbcd import CBCD,CBCDDetect,replace_detect_with_cbcd

def test_cbcd_shape_and_gradient():
 m=CBCD(16,8);x=torch.randn(2,16,15,31,requires_grad=True);y=m(x);assert y.shape==x.shape;y.mean().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

def test_cbcd_head_preserves_raw_regression():
 torch.manual_seed(7);base=YOLO('yolo11s.yaml').model.eval();target=YOLO('yolo11s.yaml').model.eval();target.load_state_dict(base.state_dict());replace_detect_with_cbcd(target)
 x=torch.randn(1,3,128,256)
 with torch.no_grad():
  oa=base(x);ob=target(x);a=oa[1] if isinstance(oa,tuple) else oa;b=ob[1] if isinstance(ob,tuple) else ob
 assert (a['boxes']-b['boxes']).abs().max()<2e-6;assert a['scores'].shape==b['scores'].shape;assert isinstance(target.model[-1],CBCDDetect)
