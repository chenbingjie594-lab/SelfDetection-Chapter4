import torch
from ultralytics.utils.nms import non_max_suppression
from chapter4_detection.scripts.stage12_candidate_analysis import traceable_nms

def test_traceable_nms_matches_ultralytics_multilabel():
    p=torch.zeros(1,6,5);p[0,:4]=torch.tensor([[10,10,30,60,90],[10,10,30,60,90],[8,8,8,8,8],[8,8,8,8,8]])
    p[0,4]=torch.tensor([.9,.8,.2,.001,0.]);p[0,5]=torch.tensor([.01,.02,.95,.8,0.])
    official,idx=non_max_suppression(p.clone(),.001,.7,nc=2,multi_label=True,max_det=300,return_idxs=True)
    traced,tids,states=traceable_nms(p.clone())
    assert torch.allclose(official[0],traced)
    assert torch.equal(idx[0],tids[:,0])
    assert any(s['reason']=='nms_suppressed' for s in states)
    assert any(s['reason']=='below_preconf' for s in states)
