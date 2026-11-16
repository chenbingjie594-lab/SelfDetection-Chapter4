import pytest,torch
from chapter4_detection.models.sfdfm import HaarDWT,HaarIDWT,SFDFM,register_sfdfm

@pytest.mark.parametrize("shape",[(2,8,16,20),(1,3,15,21)])
def test_dwt_idwt_exact_shape_and_values(shape):
 x=torch.randn(shape);bands,original=HaarDWT()(x);y=HaarIDWT()(bands,original);assert y.shape==x.shape;assert torch.allclose(x,y,atol=1e-6)
def test_fusion_shape_finite_and_initial_change():
 m=SFDFM(32,16,8).eval();x=torch.randn(2,32,17,19);y=m(x);assert y.shape==(2,16,17,19);assert torch.isfinite(y).all();assert float((y-x[:,:16]).norm()/x[:,:16].norm())<.02
def test_first_backward_reaches_every_branch():
 m=SFDFM(32,16,8).train();x=torch.randn(2,32,16,20,requires_grad=True);m(x).square().mean().backward();assert x.grad is not None
 for name,p in m.named_parameters():assert p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)>0,name
@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA AMP unavailable")
def test_amp_stability():
 m=SFDFM(32,16,8).cuda();x=torch.randn(2,32,17,19,device="cuda")
 with torch.autocast("cuda",dtype=torch.float16):y=m(x)
 assert torch.isfinite(y).all()
def test_yaml_registration_save_load(tmp_path):
 register_sfdfm();from ultralytics import YOLO
 model=YOLO("chapter4_detection/models/yolo11s_sfdfm.yaml");x=torch.randn(1,3,256,256);model.model.eval()
 with torch.no_grad():a=model.model(x)[0]
 path=tmp_path/"sfdfm.pt";model.save(path);loaded=YOLO(path);loaded.model.eval()
 with torch.no_grad():b=loaded.model(x)[0]
 assert a.shape==b.shape
