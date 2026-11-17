#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
DEVICE=${DEVICE:-0}
SEEDS=${SEEDS:-"42 3407 2026"}
S=chapter4_detection/results/stage9
DATA=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/data.yaml
IMAGES=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/images/val
LABELS=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/labels/val
mkdir -p "$S/launch_logs"
for seed in $SEEDS; do
  run=e6_cbcd_seed${seed}
  PYTHONPATH=. "$PY" chapter4_detection/scripts/train.py --experiment E6 --seed "$seed" --config chapter4_detection/configs/e6_cbcd.yaml --initialized "$S/initializations/e6_cbcd_init_seed${seed}.pt" --device "$DEVICE" 2>&1 | tee "$S/launch_logs/${run}.log"
  PYTHONPATH=. "$PY" chapter4_detection/scripts/export_validator_predictions.py --weights "$S/runs/$run/weights/best.pt" --data "$DATA" --images "$IMAGES" --output "$S/${run}_predictions.json" --imgsz 1536 --device "$DEVICE" --name "$run"
  PYTHONPATH=. "$PY" chapter4_detection/scripts/evaluate_original.py --predictions "$S/${run}_predictions.json" --labels "$LABELS" --output "$S/${run}_metrics.json"
  PYTHONPATH=. "$PY" chapter4_detection/scripts/diagnose_predictions.py --predictions "$S/${run}_predictions.json" --images "$IMAGES" --labels "$LABELS" --thresholds chapter4_detection/results/diagnostic_thresholds.json --output "$S/${run}_diagnostics.json" >/dev/null
done
