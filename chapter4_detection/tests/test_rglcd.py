import torch
from chapter4_detection.models.rglcd import RGLCD

def test_ring_statistics_exclude_padding_and_center():
 x=torch.arange(25,dtype=torch.float32).reshape(1,1,5,5);m=RGLCD(1);mean,var,d,r,count=m.statistics(x)
 vals=[]
 for yy in range(5):
  for xx in range(5):
   dy,dx=yy-2,xx-2
   if abs(dy)<=3 and abs(dx)<=3 and not(abs(dy)<=1 and abs(dx)<=1):vals.append(x[0,0,yy,xx])
 assert count[0,0,2,2].item()==len(vals)==16
 assert torch.allclose(mean[0,0,2,2],torch.stack(vals).mean())
 assert var.min()>=0 and torch.isfinite(d).all() and torch.isfinite(r).all()

def test_identity_and_staged_gradient():
 torch.manual_seed(2);m=RGLCD(4);x=torch.randn(2,4,9,11,requires_grad=True);y=m(x)
 assert torch.equal(x,y);assert torch.count_nonzero(m.phi.weight)>0
 y.square().mean().backward();assert m.gamma.grad is not None and m.gamma.grad.abs()>0;assert m.phi.weight.grad is not None and torch.count_nonzero(m.phi.weight.grad)==0
 with torch.no_grad():m.gamma.fill_(0.01)
 m.zero_grad();x.grad=None;m(x).square().mean().backward();assert torch.isfinite(m.phi.weight.grad).all() and m.phi.weight.grad.abs().sum()>0

def test_low_variance_is_finite_and_amp_compatible():
 m=RGLCD(3);x=torch.ones(1,3,8,8,dtype=torch.float16);y=m(x);assert y.dtype==x.dtype and torch.isfinite(y).all()
