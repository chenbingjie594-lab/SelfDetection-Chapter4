# Stage14 reproduction commands

Run from the repository root with the existing YOLO11 environment.

```bash
/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python -m pytest -q \
  chapter4_detection/tests/test_assignment_geometry.py \
  chapter4_detection/tests/test_assignment_monitor.py

/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python -m chapter4_detection.scripts.stage14_geometry_train \
  --data chapter4_detection/results/stage10/cv_protocol/fold0/data.yaml \
  --init chapter4_detection/results/stage7/initializations/matched_e0_init_seed42.pt \
  --project chapter4_detection/results/stage14/runs \
  --device 2

/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python -m chapter4_detection.scripts.stage14_summarize \
  --trace chapter4_detection/results/stage14/runs/e0_fold0_seed42_geometry_diagnostic/positive_geometry.csv \
  --output chapter4_detection/results/stage14/analysis
```

The training command intentionally refuses to overwrite an existing run directory. Invalid pre-runs are documented under `invalid_runs/` locally and are not evidence inputs.
