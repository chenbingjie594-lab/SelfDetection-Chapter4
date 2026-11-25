# Stage25 test results

Run date: 2026-11-25 UTC.

- Stage24 + Stage25 focused unit tests: **6 passed**.
- Python compilation passed for `models/hbd.py` and all Stage25 scripts.
- Three cache provenance checks passed: cache images exactly equal the corresponding outer-fold train list and have zero holdout overlap.
- Candidate policy checks passed: B samples have `0.05≤score<0.25`; every C candidate is excluded from training.
- Patch/full-map audit passed after the boundary-mask correction: fresh FP32 maximum absolute error ≤1.15e-5; FP16 cached mean absolute error <4e-4.
- Twelve training jobs completed 40 epochs without NaN and produced `last.pt` checkpoints.
- Twelve online YOLO integration audits passed: P3/P4/P5 box-logit changes and P4/P5 class-logit changes are exactly zero, while P3 class logits change.
- Twelve full OOF inference jobs, uniform error attribution, frozen-grid FROC/PR analysis and 5000-repeat source-group paired bootstrap completed.

CUDA remained unavailable because the NVIDIA driver could not be queried. All cached training and inference were real CPU executions; GPU memory is reported unavailable rather than fabricated.
