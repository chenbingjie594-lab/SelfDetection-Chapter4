#!/usr/bin/env bash
set -euo pipefail
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
S=chapter4_detection/results/stage25
export PYTHONPATH=.
mkdir -p "$S"/{cache,logs,runs,predictions,metrics,audits,review_plates}
for f in 0 1 2; do
  "$PY" chapter4_detection/scripts/stage25_prepare_features.py --weights "chapter4_detection/results/stage10/runs/e0_fold${f}_seed42/weights/last.pt" --images-list "chapter4_detection/results/stage10/cv_protocol/fold${f}/train.txt" --fold "$f" --output "$S/cache/fold${f}.pt"
  "$PY" chapter4_detection/scripts/stage25_patch_equivalence.py --cache "$S/cache/fold${f}.pt" --prototypes "chapter4_detection/results/stage24/prototypes/fold${f}.pt" --checkpoint "chapter4_detection/results/stage24/runs/fold${f}_matched/last.pt" --weights "chapter4_detection/results/stage10/runs/e0_fold${f}_seed42/weights/last.pt" --output "$S/audits/patch_equivalence_fold${f}.json"
  for v in H1 H2 H3 H4; do
    "$PY" chapter4_detection/scripts/stage25_train_hbd.py --cache "$S/cache/fold${f}.pt" --prototypes "chapter4_detection/results/stage24/prototypes/fold${f}.pt" --weights "chapter4_detection/results/stage10/runs/e0_fold${f}_seed42/weights/last.pt" --variant "$v" --output "$S/runs/fold${f}_${v}"
    "$PY" chapter4_detection/scripts/stage25_predict.py --weights "chapter4_detection/results/stage10/runs/e0_fold${f}_seed42/weights/last.pt" --prototypes "chapter4_detection/results/stage24/prototypes/fold${f}.pt" --checkpoint "$S/runs/fold${f}_${v}/last.pt" --images-list "chapter4_detection/results/stage10/cv_protocol/fold${f}/holdout.txt" --output "$S/predictions/fold${f}_${v}.json"
  done
done
"$PY" chapter4_detection/scripts/stage25_mine_hard_background.py --cache-dir "$S/cache" --output "$S"
"$PY" chapter4_detection/scripts/stage25_finalize.py --gt-stats chapter4_detection/results/stage10/gt_statistics.csv --labels /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/labels/train --bootstrap 5000
"$PY" chapter4_detection/scripts/stage25_feature_audit.py --gt-stats chapter4_detection/results/stage10/gt_statistics.csv
"$PY" chapter4_detection/scripts/stage25_plot_curves.py --input "$S/recall_fppi.json" --output-dir "$S"
