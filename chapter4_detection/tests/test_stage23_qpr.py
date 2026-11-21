import torch
from chapter4_detection.models.cir import apply_residual
from chapter4_detection.models.qpr_cir import QPRGate,gate_features
from chapter4_detection.scripts.stage23_common import best_rows

def test_best_candidate_can_switch():
 base={'x.jpg':{'predictions':[{'class_id':0,'xyxy':[0,0,10,10]},{'class_id':0,'xyxy':[1,1,9,9]}]}};ref={'x.jpg':{'predictions':[{'class_id':0,'xyxy':[0,0,7,7]},{'class_id':0,'xyxy':[0,0,10,10]}]}};gt=[{'image':'x.jpg','gt_id':'0','source_group':'g','class_id':'0','input_width':'2','input_height':'2','x1':'0','y1':'0','x2':'10','y2':'10'}]
 r=best_rows(base,ref,gt)[0];assert r['best_id_before']==0 and r['best_id_after']==1 and r['best_id_switched']
def test_other_candidate_preserves_coverage_when_original_best_degrades():
 base={'x.jpg':{'predictions':[{'class_id':1,'xyxy':[0,0,10,10]},{'class_id':1,'xyxy':[0,0,8,8]}]}};ref={'x.jpg':{'predictions':[{'class_id':1,'xyxy':[0,0,7,7]},{'class_id':1,'xyxy':[0,0,10,10]}]}};gt=[{'image':'x.jpg','gt_id':'0','source_group':'g','class_id':'1','input_width':'9','input_height':'9','x1':'0','y1':'0','x2':'10','y2':'10'}]
 r=best_rows(base,ref,gt)[0];assert r['fixed_candidate_iou_after']<.75 and r['best_iou_after']>=.75 and not r['lose75']
def test_gate_endpoints_and_gradient():
 box=torch.tensor([[10.,10.,20.,20.]]);d=torch.tensor([[.1,-.1,.2,-.2]]);z=torch.zeros(1,8);fh=torch.zeros(1,64);fl=torch.ones(1,64);gate=QPRGate('full');x=gate_features(z,d,fh,fl,'full');g=gate(x);assert torch.allclose(g,torch.tensor([.5]));assert torch.equal(apply_residual(box,0*d),box);assert torch.allclose(apply_residual(box,1*d),apply_residual(box,d));g.sum().backward();assert gate.net[-1].weight.grad.abs().sum()>0
