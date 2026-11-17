# Stage 9 Reproduction Commands

```bash
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
DATA=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1
PYTHONPATH=. $PY chapter4_detection/scripts/stage9_gradient_analysis.py --manifest $DATA/split_manifest.json --checkpoints CHECKPOINTS --names NAMES --output chapter4_detection/results/stage9/gradient.csv --device 0
PYTHONPATH=. $PY chapter4_detection/scripts/stage9_cbcd_audit.py --baseline BASELINE_INIT --image TRAIN_IMAGE --label TRAIN_LABEL --output CBCD_INIT --seed 42 --device 0
SEEDS="42 3407 2026" DEVICE=0 bash chapter4_detection/scripts/run_stage9_cbcd.sh
PYTHONPATH=. $PY -m pytest chapter4_detection/tests -q
```

Exact formal arguments and checkpoint hashes are retained in the scripts, audit JSON and reports. Large raw candidate tensors and checkpoints remain local and Git-ignored.
