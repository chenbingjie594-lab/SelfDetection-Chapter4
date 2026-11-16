#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
INIT=chapter4_detection/results/stage7/initializations
LOG=chapter4_detection/results/stage7/launch_logs
DATA=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/data.yaml
IMAGES=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/images/val
LABELS=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/labels/val
RUNS=chapter4_detection/results/stage7/runs
DEVICE=${DEVICE:-0}
mkdir -p "$LOG"
for seed in 42 3407 2026; do
  for spec in "M0:stage7_matched_e0:matched_e0:matched_e0" "MID:stage7_matched_identity:matched_identity:mid" "MFULL:stage7_matched_full:matched_full:mfull"; do
    IFS=: read -r experiment config run prefix <<< "$spec"
    PYTHONPATH=. "$PY" chapter4_detection/scripts/train.py --experiment "$experiment" --seed "$seed" \
      --config "chapter4_detection/configs/${config}.yaml" --initialized "$INIT/${prefix}_init_seed${seed}.pt" --device "$DEVICE" \
      2>&1 | tee "$LOG/${run}_seed${seed}.log"
    weights="$RUNS/${run}_seed${seed}/weights/best.pt"
    out="chapter4_detection/results/stage7/${run}_seed${seed}"
    PYTHONPATH=. "$PY" chapter4_detection/scripts/export_validator_predictions.py --weights "$weights" --data "$DATA" --images "$IMAGES" --output "${out}_predictions.json" --imgsz 1536 --device "$DEVICE" --name "${run}_seed${seed}"
    PYTHONPATH=. "$PY" chapter4_detection/scripts/evaluate_original.py --predictions "${out}_predictions.json" --labels "$LABELS" --output "${out}_metrics.json"
    PYTHONPATH=. "$PY" chapter4_detection/scripts/confidence_diagnostics.py --predictions "${out}_predictions.json" --labels "$LABELS" --output "${out}_confidence.json"
    PYTHONPATH=. "$PY" chapter4_detection/scripts/diagnose_predictions.py --predictions "${out}_predictions.json" --images "$IMAGES" --labels "$LABELS" --thresholds chapter4_detection/results/diagnostic_thresholds.json --output "${out}_diagnostics.json" >/dev/null
  done
done
