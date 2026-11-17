"""Classification-Branch Contrast Discrimination candidate for Stage 9."""
from __future__ import annotations
import copy
import torch
from torch import nn
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect.train import DetectionTrainer

class CBCD(nn.Module):
    """Local residual contrast enhancement used only by a classification tower."""
    def __init__(self,channels:int,hidden:int=32):
        super().__init__()
        self.smooth=nn.Conv2d(channels,channels,3,padding=1,groups=channels,bias=False)
        self.difference=nn.Sequential(nn.Conv2d(2*channels,hidden,1,bias=False),nn.BatchNorm2d(hidden),nn.SiLU())
        self.gate=nn.Sequential(nn.Conv2d(channels+hidden,hidden,1,bias=False),nn.BatchNorm2d(hidden),nn.Sigmoid())
        self.project=nn.Conv2d(hidden,channels,1,bias=False)
        with torch.no_grad():
            self.smooth.weight.fill_(1/9)
            self.project.weight.normal_(0,1e-4)
    def forward(self,x):
        residual=x-self.smooth(x);d=self.difference(torch.cat((residual,residual.abs()),1));g=self.gate(torch.cat((x,d),1));return x+self.project(g*d)

class CBCDDetect(Detect):
    """YOLO Detect head preserving box inputs and enhancing only P3 classification."""
    def __init__(self,nc=80,reg_max=16,end2end=False,ch=(),hidden=32):
        super().__init__(nc=nc,reg_max=reg_max,end2end=end2end,ch=ch);self.cbcd=CBCD(ch[0],hidden)
    def forward_head(self,x,box_head=None,cls_head=None):
        if box_head is None or cls_head is None:return {}
        bs=x[0].shape[0]
        boxes=torch.cat([box_head[i](x[i]).view(bs,4*self.reg_max,-1) for i in range(self.nl)],-1)
        cx=[self.cbcd(x[0]),*x[1:]]
        scores=torch.cat([cls_head[i](cx[i]).view(bs,self.nc,-1) for i in range(self.nl)],-1)
        return {'boxes':boxes,'scores':scores,'feats':x}

def replace_detect_with_cbcd(model,hidden=32):
    """Explicitly replace the final Detect without parser patching."""
    old=model.model[-1]
    if not isinstance(old,Detect) or isinstance(old,CBCDDetect):raise TypeError(f'Expected plain Detect, got {type(old).__name__}')
    ch=tuple(seq[0].conv.in_channels for seq in old.cv2)
    new=CBCDDetect(old.nc,old.reg_max,getattr(old,'_end2end',False),ch,hidden).to(next(old.parameters()).device)
    # Preserve non-state execution attributes (e.g. historical BN eps and
    # inplace activations) as well as tensors by copying the original towers.
    new.cv2=copy.deepcopy(old.cv2);new.cv3=copy.deepcopy(old.cv3);new.dfl=copy.deepcopy(old.dfl)
    if getattr(old,'one2one_cv2',None) is not None:
        new.one2one_cv2=copy.deepcopy(old.one2one_cv2);new.one2one_cv3=copy.deepcopy(old.one2one_cv3)
    allowed={k for k in new.state_dict() if k.startswith('cbcd.')}
    new.stride=old.stride.clone();new.f=old.f;new.i=old.i;new.type='CBCDDetect';new.np=sum(p.numel() for p in new.parameters())
    new.train(old.training)
    model.model[-1]=new
    return {'copied_keys':len(old.state_dict()),'new_keys':sorted(allowed),'channels':ch}

class CBCDTrainer(DetectionTrainer):
    """Trainer that explicitly reconstructs CBCD instead of a plain YAML Detect."""
    def get_model(self,cfg=None,weights=None,verbose=True):
        model=self.set_model_names_for_load(DetectionModel(cfg,nc=self.data['nc'],ch=self.data['channels'],verbose=verbose))
        replace_detect_with_cbcd(model,hidden=32)
        if weights:model.load(weights)
        return model
