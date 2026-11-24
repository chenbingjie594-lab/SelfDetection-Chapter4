"""Matched Background-Referenced Discrimination for YOLO11 P3 classification."""
from __future__ import annotations
import copy,torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.modules.head import Detect

class MBRD(nn.Module):
 def __init__(self,channels:int,prototypes:torch.Tensor,mode='matched',hidden=32,topk=4,tau=.1,seed=42):
  super().__init__();assert mode in ('plain','fixed','matched');self.mode=mode;self.topk=topk;self.tau=tau;self.seed=seed;self.register_buffer('prototypes',F.normalize(prototypes.float(),dim=1));self.fuse=nn.Sequential(nn.Conv2d(3*channels,hidden,1,bias=False),nn.BatchNorm2d(hidden),nn.SiLU(),nn.Conv2d(hidden,channels,1,bias=False));self.gamma=nn.Parameter(torch.zeros(()))
 def reference(self,x):
  b,c,h,w=x.shape;q=F.normalize(x.float().permute(0,2,3,1).reshape(-1,c),dim=1)
  if self.mode=='matched':
   sim=q@self.prototypes.T;v,ix=sim.topk(min(self.topk,len(self.prototypes)),dim=1);a=(v/self.tau).softmax(1);r=(self.prototypes[ix]*a[...,None]).sum(1)
  elif self.mode=='fixed':
   # Deterministic query-independent global reference, identical in training/inference.
   r=F.normalize(self.prototypes.mean(0),dim=0).expand(len(q),-1)
  else:r=q
  return r.reshape(b,h,w,c).permute(0,3,1,2).to(x.dtype)
 def forward(self,x):
  if self.mode=='plain':parts=(x,x,x)
  else:
   r=self.reference(x);parts=(x,x-r,x*r)
  h=self.fuse(torch.cat(parts,1));return x+self.gamma*h

class MBRDDetect(Detect):
 def __init__(self,nc=80,reg_max=16,end2end=False,ch=(),prototypes=None,mode='matched'):
  super().__init__(nc=nc,reg_max=reg_max,end2end=end2end,ch=ch);self.mbrd=MBRD(ch[0],prototypes,mode)
 def forward_head(self,x,box_head=None,cls_head=None):
  if box_head is None or cls_head is None:return {}
  bs=x[0].shape[0];boxes=torch.cat([box_head[i](x[i]).view(bs,4*self.reg_max,-1) for i in range(self.nl)],-1);cx=[self.mbrd(x[0]),*x[1:]];scores=torch.cat([cls_head[i](cx[i]).view(bs,self.nc,-1) for i in range(self.nl)],-1);return {'boxes':boxes,'scores':scores,'feats':x}

def replace_detect_with_mbrd(model,prototypes,mode='matched'):
 old=model.model[-1]
 if not isinstance(old,Detect) or isinstance(old,MBRDDetect):raise TypeError(type(old).__name__)
 ch=tuple(seq[0].conv.in_channels for seq in old.cv2);new=MBRDDetect(old.nc,old.reg_max,getattr(old,'_end2end',False),ch,prototypes,mode).to(next(old.parameters()).device);new.cv2=copy.deepcopy(old.cv2);new.cv3=copy.deepcopy(old.cv3);new.dfl=copy.deepcopy(old.dfl);new.stride=old.stride.clone();new.f=old.f;new.i=old.i;new.type='MBRDDetect';new.np=sum(p.numel() for p in new.parameters());new.train(old.training);model.model[-1]=new;return new
