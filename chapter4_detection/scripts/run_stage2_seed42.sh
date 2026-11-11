#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/sda/cbj/SelfDetection-main
py=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
results="$repo/chapter4_detection/results"
data=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1
slices=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1_slices_1024_o128
cd "$repo"
export PYTHONPATH="$repo"

echo "[$(date -Is)] waiting for E1 screen ch4_e1_s42"
while screen -list 2>/dev/null | grep -q '[.]ch4_e1_s42'; do sleep 30; done
grep -q 'Results saved to' "$results/launch_logs/e1_seed42.log"
test -s "$results/runs/e1_seed42/weights/best.pt"
echo "[$(date -Is)] E1 complete; starting fixed evaluation"

"$py" -m chapter4_detection.scripts.summarize_training \
  --run "$results/runs/e1_seed42" --log "$results/launch_logs/e1_seed42.log" --train-samples 138 \
  --output "$results/e1_seed42_training_summary.json"
"$py" -m chapter4_detection.scripts.evaluate_model \
  --weights "$results/runs/e1_seed42/weights/best.pt" --data "$data/data.yaml" --imgsz 1536 \
  --batch 1 --device 1 --output "$results/e1_seed42_metrics.json"
"$py" -m chapter4_detection.scripts.export_predictions \
  --weights "$results/runs/e1_seed42/weights/best.pt" --images "$data/images/val" \
  --output "$results/e1_seed42_predictions.json" --model-name E1_YOLO11s_P2_seed42 \
  --mode full --imgsz 1536 --conf 0.001 --nms-iou 0.7 --device 1
"$py" -m chapter4_detection.scripts.diagnose_predictions \
  --predictions "$results/e1_seed42_predictions.json" --images "$data/images/val" --labels "$data/labels/val" \
  --thresholds "$results/diagnostic_thresholds.json" --confidence 0.25 \
  --output "$results/e1_seed42_diagnostics.json"

echo "[$(date -Is)] starting E2 formal training"
"$py" -u -m chapter4_detection.scripts.train --experiment E2 --seed 42 \
  --config chapter4_detection/configs/e2_yolo11s_slice.yaml --data "$slices/data.yaml" --device 1 \
  2>&1 | tee "$results/launch_logs/e2_seed42.log"
test -s "$results/runs/e2_seed42/weights/last.pt"
"$py" -m chapter4_detection.scripts.summarize_training \
  --run "$results/runs/e2_seed42" --log "$results/launch_logs/e2_seed42.log" --train-samples 690 \
  --output "$results/e2_seed42_training_summary.json"

echo "[$(date -Is)] starting required E2 sliced original-image evaluation"
"$py" -m chapter4_detection.scripts.infer_sliced \
  --weights "$results/runs/e2_seed42/weights/last.pt" --images "$data/images/val" \
  --output "$results/e2_seed42_predictions_official.json" --tile-size 1024 --overlap 128 \
  --conf 0.001 --nms-iou 0.7 --device 1
"$py" -m chapter4_detection.scripts.evaluate_original \
  --predictions "$results/e2_seed42_predictions_official.json" --labels "$data/labels/val" \
  --output "$results/e2_seed42_metrics.json"

echo "[$(date -Is)] exporting unified E2 predictions and diagnostics"
"$py" -m chapter4_detection.scripts.export_predictions \
  --weights "$results/runs/e2_seed42/weights/last.pt" --images "$data/images/val" \
  --output "$results/e2_seed42_predictions.json" --model-name E2_YOLO11s_Slice_seed42 \
  --mode sliced --imgsz 1024 --tile-size 1024 --overlap 128 --conf 0.001 --nms-iou 0.7 --device 1
"$py" -m chapter4_detection.scripts.diagnose_predictions \
  --predictions "$results/e2_seed42_predictions.json" --images "$data/images/val" --labels "$data/labels/val" \
  --thresholds "$results/diagnostic_thresholds.json" --confidence 0.25 \
  --output "$results/e2_seed42_diagnostics.json"

"$py" -m chapter4_detection.scripts.visualize_diagnostics \
  --predictions "$results/e0_seed42_predictions.json" "$results/e1_seed42_predictions.json" "$results/e2_seed42_predictions.json" \
  --diagnostics "$results/e0_seed42_diagnostics.json" "$results/e1_seed42_diagnostics.json" "$results/e2_seed42_diagnostics.json" \
  --images "$data/images/val" --labels "$data/labels/val" --output "$results/qualitative"

echo "[$(date -Is)] stage2 seed42 artifacts complete"
