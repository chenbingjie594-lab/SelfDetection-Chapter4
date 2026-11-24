#!/usr/bin/env bash
set -euo pipefail
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
S=chapter4_detection/results/stage24
export PYTHONPATH=.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$S"/{cache,prototypes,runs,predictions,logs}
for fold in 0 1 2; do
  "$PY" chapter4_detection/scripts/stage24_prepare_features.py --weights "chapter4_detection/results/stage10/runs/e0_fold${fold}_seed42/weights/last.pt" --images-list "chapter4_detection/results/stage10/cv_protocol/fold${fold}/train.txt" --fold "$fold" --output "$S/cache/fold${fold}.pt"
  "$PY" chapter4_detection/scripts/stage24_build_prototypes.py --cache "$S/cache/fold${fold}.pt" --output "$S/prototypes/fold${fold}.pt" --seed 42
  for mode in plain fixed matched; do
    "$PY" chapter4_detection/scripts/stage24_train_mbrd.py --cache "$S/cache/fold${fold}.pt" --prototypes "$S/prototypes/fold${fold}.pt" --weights "chapter4_detection/results/stage10/runs/e0_fold${fold}_seed42/weights/last.pt" --mode "$mode" --output "$S/runs/fold${fold}_${mode}"
    "$PY" chapter4_detection/scripts/stage24_predict_mbrd.py --weights "chapter4_detection/results/stage10/runs/e0_fold${fold}_seed42/weights/last.pt" --prototypes "$S/prototypes/fold${fold}.pt" --checkpoint "$S/runs/fold${fold}_${mode}/last.pt" --images-list "chapter4_detection/results/stage10/cv_protocol/fold${fold}/holdout.txt" --output "$S/predictions/fold${fold}_${mode}.json"
  done
done
"$PY" chapter4_detection/scripts/stage24_finalize.py --gt-stats chapter4_detection/results/stage10/gt_statistics.csv --labels /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/labels/train --bootstrap 5000
