"""Phase-Consistent Regression Fusion: P2 geometry for P3 regression only."""
from __future__ import annotations
import copy,torch
from torch import nn
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect.train import DetectionTrainer
class PCRF(nn.Module):
 def __init__(self,p2_channels=128,p3_channels=128):
  super().__init__();self.register_buffer('smooth_kernel',torch.full((p2_channels,1,3,3),1/9));self.project=nn.Conv2d(4*p2_channels,p3_channels,1,bias=False)
  with torch.no_grad():self.project.weight.normal_(0,1e-4)
 def forward(self,p2,p3):
  r=p2-nn.functional.conv2d(p2,self.smooth_kernel,padding=1,groups=p2.shape[1]);h,w=p3.shape[-2:];r=r[...,:2*h,:2*w]
  if r.shape[-2:]!=(2*h,2*w):r=nn.functional.pad(r,(0,2*w-r.shape[-1],0,2*h-r.shape[-2]))
  return p3+self.project(nn.functional.pixel_unshuffle(r,2))
class PCRFDetect(Detect):
 def __init__(self,nc=80,reg_max=16,end2end=False,ch=(),p2_channels=128):
  super().__init__(nc,reg_max,end2end,ch);self.pcrf=PCRF(p2_channels,ch[0])
 def forward_head(self,x,box_head=None,cls_head=None):
  if box_head is None or cls_head is None:return {}
  p2,det=x[0],list(x[1:]);bs=det[0].shape[0];reg=[self.pcrf(p2,det[0]),*det[1:]]
  boxes=torch.cat([box_head[i](reg[i]).view(bs,4*self.reg_max,-1) for i in range(self.nl)],-1);scores=torch.cat([cls_head[i](det[i]).view(bs,self.nc,-1) for i in range(self.nl)],-1)
  return {'boxes':boxes,'scores':scores,'feats':det}
def replace_detect_with_pcrf(model):
 old=model.model[-1]
 if not isinstance(old,Detect) or isinstance(old,PCRFDetect):raise TypeError(type(old).__name__)
 ch=tuple(q[0].conv.in_channels for q in old.cv2);new=PCRFDetect(old.nc,old.reg_max,getattr(old,'_end2end',False),ch,128).to(next(old.parameters()).device);new.cv2=copy.deepcopy(old.cv2);new.cv3=copy.deepcopy(old.cv3);new.dfl=copy.deepcopy(old.dfl);new.stride=old.stride.clone();new.i=old.i;new.f=[2,*list(old.f)];new.type='PCRFDetect';new.np=sum(p.numel() for p in new.parameters());new.train(old.training);model.model[-1]=new;model.save=sorted(set([*model.save,2]));return new
class PCRFTrainer(DetectionTrainer):
 def get_model(self,cfg=None,weights=None,verbose=True):
  model=self.set_model_names_for_load(DetectionModel(cfg,nc=self.data['nc'],ch=self.data['channels'],verbose=verbose));replace_detect_with_pcrf(model)
  if weights:model.load(weights)
  return model
