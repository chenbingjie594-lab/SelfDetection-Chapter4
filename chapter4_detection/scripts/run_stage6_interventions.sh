#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
DATA=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/data.yaml
IMAGES=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/images/val
OUT=chapter4_detection/results/stage6/interventions
for seed in 42 3407 2026; do
  weights="chapter4_detection/results/runs/e3_seed${seed}/weights/best.pt"
  for mode in full gate_bypass residual_bypass; do
    PYTHONPATH=. "$PY" chapter4_detection/scripts/gate_intervention.py \
      --weights "$weights" --data "$DATA" --images "$IMAGES" --device 0 --seed "$seed" --mode "$mode" \
      --output "$OUT/e3_seed${seed}_${mode}_predictions.json"
  done
done
