import torch
from chapter4_detection.models.mbrd import MBRD
def test_modes_shapes_identity_and_gradient():
 x=torch.randn(2,16,8,8);p=torch.randn(32,16)
 for mode in ('plain','fixed','matched'):
  m=MBRD(16,p,mode);y=m(x);assert y.shape==x.shape and torch.equal(y,x);y.square().mean().backward();assert m.gamma.grad is not None and m.fuse[-1].weight.grad is not None
def test_matched_reference_is_vectorized_and_nearest():
 p=torch.eye(4);m=MBRD(4,p,'matched',topk=1);x=torch.zeros(1,4,1,2);x[0,0,0,0]=2;x[0,3,0,1]=3;r=m.reference(x);assert r[0,:,0,0].argmax()==0 and r[0,:,0,1].argmax()==3
def test_fixed_reference_is_query_independent():
 p=torch.randn(32,8);m=MBRD(8,p,'fixed');a=m.reference(torch.randn(1,8,4,4));b=m.reference(torch.randn(1,8,4,4));assert torch.equal(a,b)
