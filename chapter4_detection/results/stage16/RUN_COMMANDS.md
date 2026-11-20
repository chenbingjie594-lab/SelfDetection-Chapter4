# Reproduction commands

Run from the repository root with the existing YOLO11 environment:

```bash
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
DATA=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1

$PY chapter4_detection/scripts/stage16_annotation_package.py \
  --dataset "$DATA" \
  --review-table chapter4_detection/results/stage10/MANUAL_REVIEW_TABLE.csv \
  --xml-trace chapter4_detection/results/stage10/xml_trace.csv \
  --output chapter4_detection/results/stage16

$PY chapter4_detection/scripts/stage16_dataset_lineage.py \
  --dataset "$DATA" \
  --roots /mnt/sda/cbj/dataset/Plastic_Bomo /mnt/sda/cbj/dataset/PlasticBomo_detect2 \
  --output chapter4_detection/results/stage16

$PY chapter4_detection/scripts/stage16_independent_test_audit.py \
  --dataset "$DATA" \
  --roots /mnt/sda/cbj/dataset/Plastic_Bomo /mnt/sda/cbj/dataset/PlasticBomo_detect2 \
  --output chapter4_detection/results/stage16

$PY -m pytest -q chapter4_detection/tests/test_stage16_integrity.py
```

These commands are audit-only and do not invoke training.

