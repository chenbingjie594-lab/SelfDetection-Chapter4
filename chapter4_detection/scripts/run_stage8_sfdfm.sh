#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
DEVICE=${DEVICE:-0}
S=chapter4_detection/results/stage8
DATA=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/data.yaml
IMAGES=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/images/val
LABELS=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/labels/val
mkdir -p "$S/launch_logs"
for seed in 42 3407 2026; do
  run="e5_sfdfm_seed${seed}"
  PYTHONPATH=. "$PY" chapter4_detection/scripts/train.py --experiment E5 --seed "$seed" \
    --config chapter4_detection/configs/e5_sfdfm.yaml \
    --initialized "$S/initializations/e5_sfdfm_init_seed${seed}.pt" --device "$DEVICE" \
    2>&1 | tee "$S/launch_logs/${run}.log"
  out="$S/${run}"
  PYTHONPATH=. "$PY" chapter4_detection/scripts/export_validator_predictions.py --weights "$S/runs/$run/weights/best.pt" --data "$DATA" --images "$IMAGES" --output "${out}_predictions.json" --imgsz 1536 --device "$DEVICE" --name "$run"
  PYTHONPATH=. "$PY" chapter4_detection/scripts/evaluate_original.py --predictions "${out}_predictions.json" --labels "$LABELS" --output "${out}_metrics.json"
  PYTHONPATH=. "$PY" chapter4_detection/scripts/confidence_diagnostics.py --predictions "${out}_predictions.json" --labels "$LABELS" --output "${out}_confidence.json"
  PYTHONPATH=. "$PY" chapter4_detection/scripts/diagnose_predictions.py --predictions "${out}_predictions.json" --images "$IMAGES" --labels "$LABELS" --thresholds chapter4_detection/results/diagnostic_thresholds.json --output "${out}_diagnostics.json" >/dev/null
done
