"""CIR local box refiner: shared lightweight encoder and zero-init residual head."""
from __future__ import annotations
import torch
from torch import nn

class LocalEncoder(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(3,32,3,2,1,bias=False),nn.BatchNorm2d(32),nn.SiLU(),nn.Conv2d(32,64,3,2,1,bias=False),nn.BatchNorm2d(64),nn.SiLU(),nn.Conv2d(64,64,3,2,1,bias=False),nn.BatchNorm2d(64),nn.SiLU(),nn.AdaptiveAvgPool2d(1))
 def forward(self,x):return self.net(x).flatten(1)

class CIRRefiner(nn.Module):
 """Modes A=low, B=high, C=high+low+difference, C-noDiff=zeros difference."""
 def __init__(self,mode='C',semantic_dim=8):
  super().__init__();assert mode in ('A','B','C','C-noDiff');self.mode=mode;self.encoder=LocalEncoder();din=(64 if mode in ('A','B') else 192)+semantic_dim
  self.fusion=nn.Sequential(nn.Linear(din,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU());self.head=nn.Linear(64,4);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,high,low,semantic):
  if self.mode=='A':z=self.encoder(low)
  elif self.mode=='B':z=self.encoder(high)
  else:
   fh=self.encoder(high);fl=self.encoder(low);fd=torch.zeros_like(fh) if self.mode=='C-noDiff' else fh-fl;z=torch.cat((fh,fl,fd),1)
  raw=self.head(self.fusion(torch.cat((z,semantic),1)))
  # center residual <=0.5 box, log-scale residual <=log(2)
  return torch.cat((.5*torch.tanh(raw[:,:2]),.69314718056*torch.tanh(raw[:,2:])),1)

def apply_residual(boxes,d):
 cx=(boxes[:,0]+boxes[:,2])/2;cy=(boxes[:,1]+boxes[:,3])/2;w=(boxes[:,2]-boxes[:,0]).clamp_min(1e-6);h=(boxes[:,3]-boxes[:,1]).clamp_min(1e-6)
 ncx=cx+w*d[:,0];ncy=cy+h*d[:,1];nw=w*torch.exp(d[:,2]);nh=h*torch.exp(d[:,3]);return torch.stack((ncx-nw/2,ncy-nh/2,ncx+nw/2,ncy+nh/2),1)
