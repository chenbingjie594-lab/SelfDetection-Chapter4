#!/usr/bin/env bash
set -euo pipefail
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
ROOT=/mnt/sda/cbj/SelfDetection-main
cd "$ROOT"
for fold in 0 1 2; do
 "$PY" -m chapter4_detection.scripts.stage11_pcrf_cv_train --fold "$fold" --data "chapter4_detection/results/stage10/cv_protocol/fold$fold/data.yaml" --initialized chapter4_detection/results/stage11/initializations/pcrf_matched_e0_seed42.pt --project chapter4_detection/results/stage11/runs --device 0
done
