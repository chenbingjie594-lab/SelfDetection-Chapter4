import torch
from chapter4_detection.models.fspr import FSPR,phase_encoding,decode

def test_shapes_and_phase():
 x=torch.randn(2,128,8,10);p=torch.randn(2,128,4,5);m=FSPR(variant='F2');o=m(x,p)
 assert o['logits'].shape==(2,2,8,10) and o['offset'].abs().max()<=1.5 and o['edges'].min()>=0
 e=phase_encoding(4,4);assert e[0,:,0,0].tolist()==[0,0] and e[0,:,1,1].tolist()==[1,1]

def test_ablations_equal_capacity_and_distinct_inputs():
 mods=[FSPR(variant=f'F{i}') for i in range(3)];assert len({sum(p.numel() for p in m.parameters()) for m in mods})==1
 x=torch.randn(1,128,6,6);p=torch.randn(1,128,3,3);assert not torch.allclose(mods[1](x,p)['logits'],mods[2](x,p)['logits'])

def test_decode_generates_new_boxes():
 m=FSPR();o=m(torch.randn(1,128,8,8),torch.randn(1,128,4,4));o['logits'].fill_(10);r=decode(o,.05,5);assert len(r)==10 and all(x[0].shape==(4,) for x in r)

def test_gradient_all_heads_and_projections():
 m=FSPR();o=m(torch.randn(2,128,8,8),torch.randn(2,128,4,4));loss=sum(x.mean() for x in o.values());loss.backward()
 for name in ('p2.weight','p3.weight','cls.weight','off.weight','edge.weight'):assert dict(m.named_parameters())[name].grad.abs().sum()>0
