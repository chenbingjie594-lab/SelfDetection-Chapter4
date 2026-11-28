# FSPR calibration design

Outer fold-0 train groups were deterministically divided into 74 inner-fit and 18 inner-calibration images with zero source-group overlap. Per-source, per-class Platt scaling used TP IoU≥0.5, FP IoU<0.1/duplicate/class-error, and ignored ambiguous 0.1–0.5 cases. Thresholds were frozen on inner calibration.
