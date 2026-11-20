# Stage17 commands

```bash
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
DATA=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1
PYTHONPATH=chapter4_detection/scripts $PY chapter4_detection/scripts/stage17_prepare_review.py --dataset "$DATA" --stage16 chapter4_detection/results/stage16 --xml-trace chapter4_detection/results/stage10/xml_trace.csv --output chapter4_detection/results/stage17
$PY chapter4_detection/scripts/stage17_candidate_triage.py --candidates chapter4_detection/results/stage16/independent_test_candidates.csv --output chapter4_detection/results/stage17
$PY chapter4_detection/scripts/stage17_import_expert_review.py --review chapter4_detection/results/stage17/STAGE17_EXPERT_REVIEW.csv --dataset "$DATA" --output chapter4_detection/results/stage17
$PY -m pytest -q chapter4_detection/tests/test_stage17_handoff.py
```

No command trains a model or edits a source annotation.

