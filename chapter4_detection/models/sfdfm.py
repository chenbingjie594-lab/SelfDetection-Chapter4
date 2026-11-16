"""Spatial-Frequency Differential Fusion Module for Stage 8."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class HaarDWT(nn.Module):
    """Exact, fixed 2-D Haar transform with replicate padding for odd sizes."""
    def forward(self,x):
        h,w=x.shape[-2:];xp=F.pad(x,(0,w%2,0,h%2),mode="replicate")
        a,b,c,d=xp[...,0::2,0::2],xp[...,0::2,1::2],xp[...,1::2,0::2],xp[...,1::2,1::2]
        return ((a+b+c+d)/2,(-a-b+c+d)/2,(-a+b-c+d)/2,(a-b-c+d)/2),(h,w)

class HaarIDWT(nn.Module):
    def forward(self,bands,shape):
        ll,lh,hl,hh=bands
        a=(ll-lh-hl+hh)/2;b=(ll-lh+hl-hh)/2;c=(ll+lh-hl-hh)/2;d=(ll+lh+hl+hh)/2
        out=ll.new_empty(ll.shape[0],ll.shape[1],ll.shape[2]*2,ll.shape[3]*2)
        out[...,0::2,0::2]=a;out[...,0::2,1::2]=b;out[...,1::2,0::2]=c;out[...,1::2,1::2]=d
        return out[...,:shape[0],:shape[1]]

class SFDFM(nn.Module):
    """Consume aligned concatenated [P3,P4↑] and return a refined P3."""
    def __init__(self,c1,c2,branch_channels=64):
        super().__init__();self.c3,self.c4=c2,c1-c2;self.dwt=HaarDWT()
        self.p4_project=nn.Sequential(nn.Conv2d(self.c4,self.c3,1,bias=False),nn.BatchNorm2d(self.c3),nn.SiLU())
        self.detail=nn.Sequential(nn.Conv2d(3*self.c3,branch_channels,1,bias=False),nn.BatchNorm2d(branch_channels),nn.SiLU())
        self.diff=nn.Sequential(nn.Conv2d(self.c3,branch_channels,1,bias=False),nn.BatchNorm2d(branch_channels),nn.SiLU())
        self.gate=nn.Sequential(nn.Conv2d(branch_channels+2*self.c3,branch_channels,1,bias=False),nn.BatchNorm2d(branch_channels),nn.Sigmoid())
        self.fusion=nn.Conv2d(c2+2*branch_channels,c2,1,bias=False)
        with torch.no_grad():
            self.fusion.weight.zero_()
            for i in range(c2):self.fusion.weight[i,i,0,0]=1.0
            self.fusion.weight[:,c2:].normal_(0,1e-3)
        self.last_stats={}
    def forward(self,x):
        if x.shape[1]!=self.c3+self.c4:raise ValueError(f"SFDFM expects {self.c3+self.c4} channels, got {x.shape[1]}")
        f3,f4=x.split((self.c3,self.c4),dim=1);f4=self.p4_project(f4);bands,_=self.dwt(f3);h=self.detail(torch.cat(bands[1:],1));h=F.interpolate(h,size=f3.shape[-2:],mode="bilinear",align_corners=False)
        d=torch.abs(f3-f4);s=f3*f4;dc=self.diff(d);g=self.gate(torch.cat((h,d,s),1));selected=g*h
        out=self.fusion(torch.cat((f3,selected,dc),1))
        if not torch.jit.is_tracing():self.last_stats={"gate_mean":float(g.detach().mean()),"selected_detail_ratio":float(selected.norm()/(f3.norm()+1e-12))}
        return out

def register_sfdfm():
    import inspect
    import ultralytics.nn.tasks as tasks
    tasks.SFDFM=SFDFM
    if getattr(tasks.parse_model,"_sfdfm_enabled",False):return
    source=inspect.getsource(tasks.parse_model)
    marker="            A2C2f,\n"
    if marker not in source:raise RuntimeError("Unsupported Ultralytics parse_model layout for SFDFM registration")
    source=source.replace(marker,marker+"            SFDFM,\n",1)
    namespace=tasks.__dict__;exec(source,namespace);tasks.parse_model._sfdfm_enabled=True
