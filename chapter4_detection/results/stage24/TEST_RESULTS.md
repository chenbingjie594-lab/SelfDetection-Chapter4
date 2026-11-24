# Stage24 test results

Run date: 2026-11-24 (UTC).

- `python -m pytest -q chapter4_detection/tests/test_stage24_mbrd.py`: **3 passed**.
- Python compilation passed for all Stage24 model, training, prediction, evaluation and finalization scripts.
- Nine trained-model integration audits passed: P3/P4/P5 regression maximum absolute change is exactly 0; P4/P5 classification maximum absolute change is exactly 0; P3 classification changes are non-zero.
- All three prototype caches exactly match their outer-fold training lists and have zero holdout overlap.
- All 32 prototypes are retrieved in every fold.
- Nine full-image YOLO inference runs and the 5000-repeat source-group paired bootstrap completed.
- The corrected pre-NMS exporter retains all anchor-class pairs above confidence 0.001 and uses decoded `xyxy` coordinates. Formal post-NMS AP values are unchanged by this diagnostic-only correction.

Hardware note: CUDA was unavailable because the NVIDIA driver could not be queried. Cached adapter training and full-image inference were therefore executed on CPU; no result was fabricated or substituted.
