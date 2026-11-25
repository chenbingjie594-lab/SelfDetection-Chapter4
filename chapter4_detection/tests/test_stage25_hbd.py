import torch
from chapter4_detection.models.hbd import HBDProjection,hbd_loss

def test_projection_normalized_and_gradients():
    p=HBDProjection(8,32);x=torch.randn(12,8,requires_grad=True);u=p(x)
    assert torch.allclose(u.norm(dim=1),torch.ones(12),atol=1e-6)
    typ=torch.tensor([0,0,2,2,0,0,2,2,1,1,1,1]);cls=torch.tensor([0,0,0,0,1,1,1,1,-1,-1,-1,-1]);w=torch.where(typ==2,.25,1.)
    loss,valid=hbd_loss(u,typ,cls,w);loss.backward();assert valid and p.proj.weight.grad.abs().sum()>0

def test_unconditional_control():
    u=torch.nn.functional.normalize(torch.randn(8,32),dim=1);typ=torch.tensor([0,0,0,0,2,2,2,2]);cls=torch.tensor([0,1,0,1,0,1,0,1]);w=torch.ones(8)
    loss,valid=hbd_loss(u,typ,cls,w,class_conditional=False);assert torch.isfinite(loss) and valid[0]['class']==-1

def test_missing_class_is_safe():
    u=torch.nn.functional.normalize(torch.randn(3,32),dim=1);typ=torch.tensor([0,1,1]);cls=torch.tensor([0,-1,-1]);w=torch.ones(3)
    loss,valid=hbd_loss(u,typ,cls,w);assert loss.item()==0 and valid==[]
