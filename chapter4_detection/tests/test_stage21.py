import numpy as np
from chapter4_detection.scripts.stage21_annotation_sensitivity import perturb
from chapter4_detection.scripts.stage21_scale_learnability import corrected,iou

def test_perturbation_is_legal_and_deterministic():
 b=np.array([0.,0.,2.,2.]);a=perturb(b,'independent',2,np.random.default_rng(42),4,4);q=perturb(b,'independent',2,np.random.default_rng(42),4,4)
 assert np.array_equal(a,q) and a[2]>a[0] and a[3]>a[1]

def test_log_scale_correction():
 b=np.array([0.,0.,10.,10.]);q=corrected(b,np.log([1.5,.8]))
 assert np.allclose(q,[-2.5,1.,12.5,9.]) and np.isclose(iou(b,b),1.0)
