#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/cbj/SelfDetection-main
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
S=chapter4_detection/results/stage22
export PYTHONPATH=.
echo "[$(date -u +%FT%TZ)] Stage22 remaining run started"

for pair in "B B" "C C" "C-noDiff C_noDiff"; do
  read -r variant name <<< "$pair"
  "$PY" chapter4_detection/scripts/stage22_train_cir.py --candidates "$S/candidates/fold2_train.json" --variant "$variant" --output "$S/runs/fold2_$name" --epochs 40 --batch 32 --device cpu --seed 42
done

for fold in 0 1 2; do
  "$PY" chapter4_detection/scripts/stage22_prepare_candidates.py --images-list "chapter4_detection/results/stage10/cv_protocol/fold$fold/holdout.txt" --frozen-predictions "chapter4_detection/results/stage10/e0_fold${fold}_seed42_predictions.json" --mode eval --fold "$fold" --output "$S/candidates/fold${fold}_eval.json"
  for pair in "A A" "B B" "C C" "C-noDiff C_noDiff"; do
    read -r variant name <<< "$pair"
    "$PY" chapter4_detection/scripts/stage22_predict_cir.py --candidates "$S/candidates/fold${fold}_eval.json" --checkpoint "$S/runs/fold${fold}_$name/last.pt" --output "$S/predictions/fold${fold}_$name.json" --device cpu
    "$PY" chapter4_detection/scripts/evaluate_original.py --predictions "$S/predictions/fold${fold}_$name.json" --labels /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/labels/train --output "$S/metrics/fold${fold}_${name}_ap.json"
    "$PY" chapter4_detection/scripts/stage22_evaluate.py --base "chapter4_detection/results/stage10/e0_fold${fold}_seed42_predictions.json" --refined "$S/predictions/fold${fold}_$name.json" --gt-stats chapter4_detection/results/stage10/gt_statistics.csv --images-list "chapter4_detection/results/stage10/cv_protocol/fold$fold/holdout.txt" --output "$S/metrics/fold${fold}_${name}_candidate.json"
  done
done
echo "[$(date -u +%FT%TZ)] Stage22 remaining run completed"
