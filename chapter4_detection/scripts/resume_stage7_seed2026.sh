#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
DEVICE=${DEVICE:-0}
S=chapter4_detection/results/stage7
RUNS=$S/runs
DATA=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/data.yaml
IMAGES=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/images/val
LABELS=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/labels/val

evaluate_run() {
  local run=$1
  local out="$S/${run}"
  PYTHONPATH=. "$PY" chapter4_detection/scripts/export_validator_predictions.py \
    --weights "$RUNS/$run/weights/best.pt" --data "$DATA" --images "$IMAGES" \
    --output "${out}_predictions.json" --imgsz 1536 --device "$DEVICE" --name "$run"
  PYTHONPATH=. "$PY" chapter4_detection/scripts/evaluate_original.py \
    --predictions "${out}_predictions.json" --labels "$LABELS" --output "${out}_metrics.json"
  PYTHONPATH=. "$PY" chapter4_detection/scripts/confidence_diagnostics.py \
    --predictions "${out}_predictions.json" --labels "$LABELS" --output "${out}_confidence.json"
  PYTHONPATH=. "$PY" chapter4_detection/scripts/diagnose_predictions.py \
    --predictions "${out}_predictions.json" --images "$IMAGES" --labels "$LABELS" \
    --thresholds chapter4_detection/results/diagnostic_thresholds.json \
    --output "${out}_diagnostics.json" >/dev/null
}

# The original process ended during epoch 82; last.pt is the complete epoch-81 state.
PYTHONPATH=. DEVICE="$DEVICE" "$PY" -c \
  'import os; from ultralytics import YOLO; YOLO("chapter4_detection/results/stage7/runs/matched_e0_seed2026/weights/last.pt").train(resume=True, device=os.environ["DEVICE"])' \
  2>&1 | tee -a "$S/launch_logs/matched_e0_seed2026.log"
evaluate_run matched_e0_seed2026

for spec in "MID:stage7_matched_identity:matched_identity:mid" "MFULL:stage7_matched_full:matched_full:mfull"; do
  IFS=: read -r experiment config run prefix <<< "$spec"
  PYTHONPATH=. "$PY" chapter4_detection/scripts/train.py --experiment "$experiment" --seed 2026 \
    --config "chapter4_detection/configs/${config}.yaml" \
    --initialized "$S/initializations/${prefix}_init_seed2026.pt" --device "$DEVICE" \
    2>&1 | tee "$S/launch_logs/${run}_seed2026.log"
  evaluate_run "${run}_seed2026"
done
