import torch
from ultralytics import YOLO
from chapter4_detection.models.pcrf import PCRFDetect,replace_detect_with_pcrf
def test_pcrf_shapes_and_classification_identity():
 m=YOLO('yolo11s.yaml').model.eval();old=m.model[-1];p3=torch.randn(1,128,16,40);p4=torch.randn(1,256,8,20);p5=torch.randn(1,512,4,10);p2=torch.randn(1,128,32,80)
 with torch.no_grad():c=old.cv3[0](p3).clone()
 new=replace_detect_with_pcrf(m)
 with torch.no_grad():out=new.forward_head([p2,p3,p4,p5],**new.one2many)
 assert out['boxes'].shape[-1]==16*40+8*20+4*10
 assert torch.equal(c,new.cv3[0](p3))
 assert new.pcrf.project.weight.grad is None
def test_pcrf_gradient():
 m=YOLO('yolo11s.yaml').model;new=replace_detect_with_pcrf(m);new.train();xs=[torch.randn(1,128,32,80),torch.randn(1,128,16,40),torch.randn(1,256,8,20),torch.randn(1,512,4,10)];out=new.forward_head(xs,**new.one2many);out['boxes'].sum().backward();g=new.pcrf.project.weight.grad;assert g is not None and torch.isfinite(g).all() and g.norm()>0
