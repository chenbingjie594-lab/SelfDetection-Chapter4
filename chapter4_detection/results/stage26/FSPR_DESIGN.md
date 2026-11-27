# FSPR design

FSPR is a 9,032-parameter frozen-E0 sidecar. It projects backbone P2 and neck P3 to 24 channels, nearest-aligns P3 at stride 4, optionally adds the explicit within-2×2 phase residual and fixed phase coordinates, and predicts two heatmap classes, bounded center offsets and four positive edge distances. F0/F1/F2 have identical capacity through zero-input controls. Original Detect logits, boxes and NMS outputs are never modified.
