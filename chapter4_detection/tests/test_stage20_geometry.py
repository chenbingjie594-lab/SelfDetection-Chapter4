import numpy as np
import torch

from chapter4_detection.scripts.stage20_geometry import geom, iou, repair
from chapter4_detection.scripts.stage20_p3_boundary_analysis import sample


def test_geometry_and_repairs():
    gt=(0.,0.,10.,10.); pred=(-1.,1.,8.,9.)
    r=repair(gt,pred)
    assert r['size']>iou(gt,pred)
    assert r['center']>iou(gt,pred)
    assert r['lr']>=r['l'] and r['tb']>=r['t']
    g=geom(gt,pred)
    assert g['left_orig_px']==-1 and g['right_orig_px']==-2
    assert np.isclose(g['center_x_input_px'],.5625)
    assert 0 <= iou(gt,pred) <= 1


def test_bilinear_sample_align_corners_false():
    feat=torch.arange(16,dtype=torch.float32).reshape(1,1,4,4)
    assert torch.allclose(sample(feat,1.,2.),torch.tensor([9.]))
    assert torch.allclose(sample(feat,1.5,1.5),torch.tensor([7.5]))
