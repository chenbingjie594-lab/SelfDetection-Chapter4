"""Reliability-Gated Local Contrast Discriminator for YOLO11 P3 classification."""
from __future__ import annotations
import copy
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect.train import DetectionTrainer

class RGLCD(nn.Module):
    """Fixed 7x7 minus 3x3 ring statistics and a zero-gated learned projection."""
    def __init__(self, channels:int, eps:float=1e-4, alpha:float=1.0, diagnostics_enabled:bool=False):
        super().__init__(); self.eps=float(eps); self.alpha=float(alpha)
        mask=torch.ones(1,1,7,7); mask[:,:,2:5,2:5]=0
        self.register_buffer('ring_mask',mask)
        self.phi=nn.Conv2d(channels,channels,1,bias=False)
        nn.init.kaiming_normal_(self.phi.weight,mode='fan_out',nonlinearity='linear')
        self.gamma=nn.Parameter(torch.zeros(())); self.diagnostics_enabled=bool(diagnostics_enabled)
        self.last_stats={}
    def statistics(self,x):
        xf=x.float(); c=xf.shape[1]; k=self.ring_mask.float().expand(c,1,7,7)
        valid=torch.ones_like(xf[:,:1]); count=F.conv2d(valid,self.ring_mask.float(),padding=3).clamp_min(1.0)
        mean=F.conv2d(xf,k,padding=3,groups=c)/count
        second=F.conv2d(xf.square(),k,padding=3,groups=c)/count
        var=(second-mean.square()).clamp_min(0.0)
        d=(xf-mean)*torch.rsqrt(var+self.eps); r=torch.exp(-self.alpha*var)
        return mean,var,d,r,count
    def forward(self,x):
        mean,var,d,r,count=self.statistics(x)
        residual=r*self.phi(d.to(self.phi.weight.dtype)).float()
        if getattr(self,'diagnostics_enabled',False) and not torch.jit.is_scripting():
            self.last_stats={'variance_mean':float(var.detach().mean()),'reliability_mean':float(r.detach().mean()),'contrast_abs_mean':float(d.detach().abs().mean()),'residual_abs_mean':float(residual.detach().abs().mean()),'valid_min':float(count.detach().min()),'valid_max':float(count.detach().max())}
        return (x.float()+self.gamma.float()*residual).to(x.dtype)

class RGLCDDetect(Detect):
    def __init__(self,nc=80,reg_max=16,end2end=False,ch=(),eps=1e-4,alpha=1.0):
        super().__init__(nc=nc,reg_max=reg_max,end2end=end2end,ch=ch);self.rglcd=RGLCD(ch[0],eps,alpha)
    def forward_head(self,x,box_head=None,cls_head=None):
        if box_head is None or cls_head is None:return {}
        bs=x[0].shape[0];boxes=torch.cat([box_head[i](x[i]).view(bs,4*self.reg_max,-1) for i in range(self.nl)],-1)
        cx=[self.rglcd(x[0]),*x[1:]];scores=torch.cat([cls_head[i](cx[i]).view(bs,self.nc,-1) for i in range(self.nl)],-1)
        return {'boxes':boxes,'scores':scores,'feats':x}

def replace_detect_with_rglcd(model,eps=1e-4,alpha=1.0):
    old=model.model[-1]
    if not isinstance(old,Detect) or isinstance(old,RGLCDDetect):raise TypeError(f'Expected plain Detect, got {type(old).__name__}')
    ch=tuple(seq[0].conv.in_channels for seq in old.cv2);new=RGLCDDetect(old.nc,old.reg_max,getattr(old,'_end2end',False),ch,eps,alpha).to(next(old.parameters()).device)
    new.cv2=copy.deepcopy(old.cv2);new.cv3=copy.deepcopy(old.cv3);new.dfl=copy.deepcopy(old.dfl)
    if getattr(old,'one2one_cv2',None) is not None:new.one2one_cv2=copy.deepcopy(old.one2one_cv2);new.one2one_cv3=copy.deepcopy(old.one2one_cv3)
    new.stride=old.stride.clone();new.f=old.f;new.i=old.i;new.type='RGLCDDetect';new.np=sum(p.numel() for p in new.parameters());new.train(old.training);model.model[-1]=new
    return {'copied_keys':len(old.state_dict()),'new_keys':sorted(k for k in new.state_dict() if k.startswith('rglcd.')),'channels':ch}

class RGLCDTrainer(DetectionTrainer):
    def get_model(self,cfg=None,weights=None,verbose=True):
        model=self.set_model_names_for_load(DetectionModel(cfg,nc=self.data['nc'],ch=self.data['channels'],verbose=verbose));replace_detect_with_rglcd(model)
        if weights:model.load(weights)
        return model
