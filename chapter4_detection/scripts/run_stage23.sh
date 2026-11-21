#!/usr/bin/env bash
set -euo pipefail
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
S=chapter4_detection/results/stage23
GT=chapter4_detection/results/stage10/gt_statistics.csv
mkdir -p "$S"
export PYTHONPATH=.
"$PY" chapter4_detection/scripts/stage23_correct_stage22.py --gt-stats "$GT" --output "$S"
for fold in 0 1 2; do
  "$PY" chapter4_detection/scripts/stage23_crossfit_cir.py --outer-fold "$fold" --candidates "chapter4_detection/results/stage22/candidates/fold${fold}_train.json" --gt-stats "$GT" --outer-holdout "chapter4_detection/results/stage10/cv_protocol/fold${fold}/holdout.txt" --output "$S/crossfit/fold${fold}" --python "$PY"
  for mode in geo full; do
    "$PY" chapter4_detection/scripts/stage23_train_gate.py --targets "$S/crossfit/fold${fold}/quality_targets.json" --mode "$mode" --output "$S/runs/fold${fold}_${mode}"
  done
  "$PY" chapter4_detection/scripts/stage23_predict_gate.py --candidates "chapter4_detection/results/stage22/candidates/fold${fold}_eval.json" --cir-checkpoint "chapter4_detection/results/stage22/runs/fold${fold}_C/last.pt" --fixed-gate .5 --output "$S/predictions/fold${fold}_half.json"
  for mode in geo full; do
    "$PY" chapter4_detection/scripts/stage23_predict_gate.py --candidates "chapter4_detection/results/stage22/candidates/fold${fold}_eval.json" --cir-checkpoint "chapter4_detection/results/stage22/runs/fold${fold}_C/last.pt" --gate-checkpoint "$S/runs/fold${fold}_${mode}/last.pt" --output "$S/predictions/fold${fold}_${mode}.json"
  done
done
