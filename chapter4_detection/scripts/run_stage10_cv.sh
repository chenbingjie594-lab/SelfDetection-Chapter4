#!/usr/bin/env bash
set -euo pipefail
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
ROOT=/mnt/sda/cbj/SelfDetection-main
CV=$ROOT/chapter4_detection/results/stage10/cv_protocol
RUNS=$ROOT/chapter4_detection/results/stage10/runs
E0=$ROOT/chapter4_detection/results/stage7/initializations/matched_e0_init_seed42.pt
CBCD=$ROOT/chapter4_detection/results/stage10/initializations/cbcd_matched_e0_seed42.pt
cd "$ROOT"
for fold in 0 1 2; do
  for model in e0 cbcd; do
    "$PY" -m chapter4_detection.scripts.stage10_cv_train --model "$model" --fold "$fold" --data "$CV/fold$fold/data.yaml" --seed 42 --device 0 --project "$RUNS" --e0-init "$E0" --cbcd-init "$CBCD"
  done
done
