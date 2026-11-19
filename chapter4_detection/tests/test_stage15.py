import numpy as np
from chapter4_detection.scripts.stage15_resolution_audit import crop_box, nonoverlap_background, edge_peak_shift

def test_background_roi_is_same_size_and_nonoverlap():
 gt=(100.,100.,120.,120.);b=nonoverlap_background(gt,4096,1024,60,60)
 assert b[2]-b[0]==60 and b[3]-b[1]==60
 assert min(b[2],gt[2])<=max(b[0],gt[0]) or min(b[3],gt[3])<=max(b[1],gt[1])

def test_crop_box_stays_in_image():
 assert crop_box(0,0,32,32,4096,1024)==(0,0,32,32)
 assert crop_box(4096,1024,32,32,4096,1024)==(4064,992,4096,1024)

def test_edge_peak_shift_zero_for_identical_images():
 x=np.zeros((32,32,3),np.uint8);x[:,16:]=255
 assert edge_peak_shift(x,x)==0
