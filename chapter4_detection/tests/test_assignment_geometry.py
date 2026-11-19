import torch
from chapter4_detection.models.assignment_geometry_monitor import representable_iou_max
from ultralytics.utils.tal import bbox2dist, dist2bbox

def test_representable_iou_inside_is_one():
 assert abs(representable_iou_max((5,5),(0,0,10,10),(100,100))-1)<1e-12
def test_representable_iou_external_anchor_enclosing_box():
 # Best legal box spans anchor x=-5 through GT right=10: IoU=100/150.
 assert abs(representable_iou_max((-5,5),(0,0,10,10),(100,100))-2/3)<1e-12
def test_representable_iou_distance_cap():
 assert 0<=representable_iou_max((-20,5),(0,0,10,10),(15,15))<.5

def test_installed_bbox2dist_preserves_signed_then_clips_to_1499():
 anchor=torch.tensor([[[-2.,5.]]]);box=torch.tensor([[[0.,0.,20.,10.]]])
 raw=bbox2dist(anchor,box,None);actual=bbox2dist(anchor,box,15)
 assert torch.equal(raw,torch.tensor([[[-2.,5.,22.,5.]]]))
 assert torch.allclose(actual,torch.tensor([[[0.,5.,14.99,5.]]]),atol=1e-6)

def test_zero_endpoint_is_closest_decodable_box_for_external_anchor():
 anchor=torch.tensor([[-5.,5.]])
 distance=torch.tensor([[0.,5.,15.,5.]])
 decoded=dist2bbox(distance,anchor,xywh=False)
 assert torch.equal(decoded,torch.tensor([[-5.,0.,10.,10.]]))
 assert abs(representable_iou_max((-5,5),(0,0,10,10),(100,100))-2/3)<1e-12
