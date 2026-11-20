# Stage21 reproduction

```bash
export CH4_PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
PYTHONPATH=. $CH4_PY chapter4_detection/scripts/stage21_annotation_sensitivity.py --predictions chapter4_detection/results/stage10/e0_oof_seed42_predictions.json --gt-stats chapter4_detection/results/stage10/gt_statistics.csv --output chapter4_detection/results/stage21
PYTHONPATH=. $CH4_PY chapter4_detection/scripts/stage21_scale_learnability.py --features chapter4_detection/results/stage12/candidate_quality_features.csv --gt-stats chapter4_detection/results/stage10/gt_statistics.csv --train-list chapter4_detection/results/stage10/cv_protocol/fold0/train.txt --output chapter4_detection/results/stage21 --bootstrap 5000
PYTHONPATH=. $CH4_PY chapter4_detection/scripts/stage21_finalize.py --output chapter4_detection/results/stage21 --gt-stats chapter4_detection/results/stage10/gt_statistics.csv --geometry chapter4_detection/results/stage20/geometry_by_gt_selection.csv --xml-trace chapter4_detection/results/stage10/xml_trace.csv --expert-review chapter4_detection/results/stage17/STAGE17_EXPERT_REVIEW.csv
PYTHONPATH=. $CH4_PY -m pytest -q chapter4_detection/tests/test_stage21.py
```
