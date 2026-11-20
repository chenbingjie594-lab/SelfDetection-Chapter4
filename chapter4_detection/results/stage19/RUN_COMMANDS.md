# Stage19 commands

```bash
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
nvidia-smi
$PY -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
PYTHONPATH=. $PY chapter4_detection/scripts/stage19_feature_audit.py --init chapter4_detection/results/stage18/initializations/rglcd_matched_e0_seed42.pt --trained-e0 chapter4_detection/results/stage10/runs/e0_fold0_seed42/weights/last.pt --train-list chapter4_detection/results/stage10/cv_protocol/fold0/train.txt --gt-stats chapter4_detection/results/stage10/gt_statistics.csv --output chapter4_detection/results/stage19
PYTHONPATH=. $PY chapter4_detection/scripts/stage19_visualize_hard.py --instances chapter4_detection/results/stage19/stage19_feature_instances.csv --train-list chapter4_detection/results/stage10/cv_protocol/fold0/train.txt --output chapter4_detection/results/stage19/hard_background_review --count 12
PYTHONPATH=. $PY -m pytest -q chapter4_detection/tests/test_rglcd.py chapter4_detection/tests/test_stage19_audit.py
```

Formal training is intentionally absent because `FAIL_MECHANISM_RECHECK` closed the route.
