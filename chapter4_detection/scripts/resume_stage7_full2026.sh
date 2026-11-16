#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
DEVICE=${DEVICE:-0}
S=chapter4_detection/results/stage7
RUN=matched_full_seed2026
DATA=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/data.yaml
IMAGES=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/images/val
LABELS=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/labels/val

PYTHONPATH=. DEVICE="$DEVICE" "$PY" -c \
  'import os; from chapter4_detection.models.nbr_de import register_nbr_de; register_nbr_de(); from ultralytics import YOLO; YOLO("chapter4_detection/results/stage7/runs/matched_full_seed2026/weights/last.pt").train(resume=True, device=os.environ["DEVICE"])' \
  2>&1 | tee -a "$S/launch_logs/${RUN}.log"
PYTHONPATH=. "$PY" chapter4_detection/scripts/export_validator_predictions.py \
  --weights "$S/runs/$RUN/weights/best.pt" --data "$DATA" --images "$IMAGES" \
  --output "$S/${RUN}_predictions.json" --imgsz 1536 --device "$DEVICE" --name "$RUN"
PYTHONPATH=. "$PY" chapter4_detection/scripts/evaluate_original.py \
  --predictions "$S/${RUN}_predictions.json" --labels "$LABELS" --output "$S/${RUN}_metrics.json"
PYTHONPATH=. "$PY" chapter4_detection/scripts/confidence_diagnostics.py \
  --predictions "$S/${RUN}_predictions.json" --labels "$LABELS" --output "$S/${RUN}_confidence.json"
PYTHONPATH=. "$PY" chapter4_detection/scripts/diagnose_predictions.py \
  --predictions "$S/${RUN}_predictions.json" --images "$IMAGES" --labels "$LABELS" \
  --thresholds chapter4_detection/results/diagnostic_thresholds.json \
  --output "$S/${RUN}_diagnostics.json" >/dev/null
