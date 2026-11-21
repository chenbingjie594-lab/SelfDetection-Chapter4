import numpy as np,torch
from chapter4_detection.models.cir import CIRRefiner,apply_residual
from chapter4_detection.scripts.stage22_common import low_reconstruction,roi,crop,target_residual
def test_zero_init_and_gradients():
 for mode in ('A','B','C','C-noDiff'):
  m=CIRRefiner(mode);h=torch.rand(2,3,128,128);l=torch.rand_like(h);s=torch.rand(2,8);d=m(h,l,s);assert torch.equal(d,torch.zeros_like(d));loss=(d-torch.ones_like(d)*.1).square().mean();loss.backward();assert m.head.weight.grad.abs().sum()>0
def test_geometry_and_alignment():
 im=np.arange(64*256*3,dtype=np.uint8).reshape(64,256,3);lo=low_reconstruction(im);r,side=roi([10,10,20,20]);assert side==96 and crop(im,r).shape==crop(lo,r).shape==(128,128,3)
 b=np.array([10.,10.,20.,20.]);g=np.array([11.,9.,23.,21.]);t=target_residual(b,g);q=apply_residual(torch.tensor(b).view(1,4),torch.tensor(t).view(1,4)).numpy()[0];assert np.allclose(q,g,atol=1e-5)
def test_save_reload(tmp_path):
 m=CIRRefiner('C');p=tmp_path/'x.pt';torch.save({'variant':'C','state_dict':m.state_dict()},p);q=torch.load(p,weights_only=False);n=CIRRefiner(q['variant']);n.load_state_dict(q['state_dict']);assert all(torch.equal(a,b) for a,b in zip(m.state_dict().values(),n.state_dict().values()))
