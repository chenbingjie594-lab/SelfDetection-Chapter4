"""Quality-preserving gate for a frozen CIR box residual."""
from __future__ import annotations
import torch
from torch import nn

GEO_DIM=13
FULL_DIM=205

class QPRGate(nn.Module):
 def __init__(self,mode="geo"):
  super().__init__();assert mode in ("geo","full");self.mode=mode;din=GEO_DIM if mode=="geo" else FULL_DIM
  self.net=nn.Sequential(nn.Linear(din,32),nn.SiLU(),nn.Linear(32,1))
  # Start conservatively at half refinement, without blocking gradients.
  nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,x):return torch.sigmoid(self.net(x)).squeeze(1)

def gate_features(semantic,residual,fh=None,fl=None,mode="geo"):
 geo=torch.cat((semantic,residual,residual.norm(dim=1,keepdim=True)),1)
 if mode=="geo":return geo
 if fh is None or fl is None:raise ValueError("full gate requires high/low CIR features")
 return torch.cat((geo,fh,fl,(fh-fl).abs()),1)
