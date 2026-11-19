# TAL–DFL geometric audit

## Audited implementation

Environment: Ultralytics 8.4.145, YOLO11s, `reg_max=16`, P3/P4/P5 strides 8/16/32.

The installed code establishes the following chain.

1. `TaskAlignedAssigner.select_candidates_in_gts` floors each GT side below 16 input pixels to 16 **only while constructing the candidate mask**.
2. `get_targets` gathers `target_bboxes` from the unchanged `gt_bboxes`; the expanded box is not a regression target.
3. For anchor (a=(a_x,a_y)) and original GT (B=(x_1,y_1,x_2,y_2)), `bbox2dist` forms ((a_x-x_1,a_y-y_1,x_2-a_x,y_2-a_y)).
4. `DFLoss` clips each component in-place to `[0, 14.99]` (`reg_max - 1 - 0.01`) and linearly interpolates cross entropy between the adjacent integer bins.
5. `dist2bbox` decodes non-negative ltrb distances around the anchor.
6. `BboxLoss` computes CIoU against the original GT. It does not use the candidate-expansion box.

All statements above were checked against the local files `ultralytics/utils/tal.py` and `ultralytics/utils/loss.py`; no site-package was modified.

## Geometry

The decoded box must contain its anchor center. If an anchor is outside the GT, the GT itself is therefore not exactly representable. The closest legal edge is distance zero, so clipping a negative signed distance to zero is the projection of that edge onto the legal DFL domain. Likewise, the upper endpoint is the closest legal edge when a target exceeds the maximum distance.

For independent axis-aligned edges, this gives an exact maximum-IoU construction: choose each reachable predicted edge closest to the corresponding GT edge, then compute IoU of that box with the GT. `representable_iou_max` implements this construction and is covered by inside-anchor, outside-anchor, and finite-range tests.

## Does DFL conflict with CIoU?

There is a parameterization constraint, but no demonstrated directional contradiction. For an exterior anchor, both the clipped DFL endpoint and the constrained IoU optimum place the invalid edge at distance zero. CIoU may still provide center/aspect gradients, but it cannot make a non-negative-distance decoder place an edge beyond the anchor in the forbidden direction.

Thus “negative target was clipped” is not itself a training bug. It identifies a geometrically limited positive, not an incorrect implementation.
