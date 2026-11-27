"""Fine-grid semantic-guided proposal recovery sidecar."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def phase_encoding(h:int,w:int,device=None,dtype=None,origin=(0,0)):
 y=(torch.arange(h,device=device)+origin[0])%2;x=(torch.arange(w,device=device)+origin[1])%2
 yy,xx=torch.meshgrid(y,x,indexing='ij');return torch.stack((xx,yy)).to(dtype=dtype)[None]

class FSPR(nn.Module):
 def __init__(self,c2=128,c3=128,width=24,variant='F2'):
  super().__init__();assert variant in ('F0','F1','F2');self.variant=variant;self.width=width
  self.p2=nn.Conv2d(c2,width,1);self.p3=nn.Conv2d(c3,width,1)
  cin=3*width+2;self.fuse=nn.Sequential(nn.Conv2d(cin,width,1),nn.SiLU(),nn.Conv2d(width,width,3,padding=1,groups=width),nn.SiLU(),nn.Conv2d(width,width,1),nn.SiLU())
  self.cls=nn.Conv2d(width,2,1);self.off=nn.Conv2d(width,2,1);self.edge=nn.Conv2d(width,4,1)
  nn.init.constant_(self.cls.bias,-4.595)
 def align(self,f2,f3):return f2,F.interpolate(f3,size=f2.shape[-2:],mode='nearest')
 def forward(self,f2,f3,valid_mask=None,origin=(0,0)):
  q2=self.p2(f2);q3=self.p3(f3) if f3.shape[-2:]==f2.shape[-2:] else F.interpolate(self.p3(f3),size=f2.shape[-2:],mode='nearest')
  if valid_mask is not None:q2=q2*valid_mask[:,None];q3=q3*valid_mask[:,None]
  py,px=origin[0]%2,origin[1]%2;zpad=F.pad(q2,(px,(q2.shape[-1]+px)%2,py,(q2.shape[-2]+py)%2));pooled=F.avg_pool2d(zpad,2,2);mean=F.interpolate(pooled,scale_factor=2,mode='nearest')[...,py:py+q2.shape[-2],px:px+q2.shape[-1]];d=q2-mean
  if self.variant=='F0':q2=torch.zeros_like(q2);d=torch.zeros_like(d)
  elif self.variant=='F1':d=torch.zeros_like(d)
  e=phase_encoding(q2.shape[-2],q2.shape[-1],q2.device,q2.dtype,origin).expand(len(q2),-1,-1,-1)
  z=torch.cat((q2,q3,d,e),1)
  if valid_mask is not None:z=z*valid_mask[:,None]
  for layer in self.fuse:
   z=layer(z)
   if valid_mask is not None and isinstance(layer,(nn.Conv2d,nn.SiLU)):z=z*valid_mask[:,None]
  return {'logits':self.cls(z),'offset':1.5*torch.tanh(self.off(z)),'edges':F.softplus(self.edge(z))}

def decode(out,score_min=.05,topk=100):
 """Return dense input-space xyxy, scores and classes before source merging."""
 s=out['logits'].sigmoid()[0];off=out['offset'][0];ed=out['edges'][0];h,w=s.shape[-2:];yy,xx=torch.meshgrid(torch.arange(h,device=s.device),torch.arange(w,device=s.device),indexing='ij');cx=xx+.5+off[0];cy=yy+.5+off[1];box=torch.stack((cx-ed[0],cy-ed[1],cx+ed[2],cy+ed[3]),-1)*4;rows=[]
 for c in range(2):
  ids=torch.where(s[c].flatten()>=score_min)[0];ids=ids[s[c].flatten()[ids].argsort(descending=True)[:topk]]
  for j in ids:rows.append((box.reshape(-1,4)[j],s[c].flatten()[j],c,j))
 return rows
