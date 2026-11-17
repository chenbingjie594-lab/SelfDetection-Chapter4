# Stage 8 Reproduction Commands

Run from the repository root in the existing YOLO11 environment.

```bash
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
DATA=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1
PYTHONPATH=. $PY chapter4_detection/scripts/stage8_frequency_analysis.py --manifest $DATA/split_manifest.json --images $DATA/images/train --labels $DATA/labels/train --output-dir chapter4_detection/results/stage8
PYTHONPATH=. $PY chapter4_detection/scripts/stage8_feature_diagnosis.py --manifest $DATA/split_manifest.json --images $DATA/images/train --labels $DATA/labels/train --output-dir chapter4_detection/results/stage8 --device 0
PYTHONPATH=. $PY chapter4_detection/scripts/stage8_model_audit.py --device 0
DEVICE=0 bash chapter4_detection/scripts/run_stage8_sfdfm.sh
PYTHONPATH=. $PY chapter4_detection/scripts/stage8_paired_bootstrap.py --manifest $DATA/split_manifest.json --labels $DATA/labels/val --baseline-dir chapter4_detection/results/stage7 --sfdfm-dir chapter4_detection/results/stage8 --output chapter4_detection/results/stage8/paired_group_bootstrap.json --repeats 5000
PYTHONPATH=. $PY -m pytest chapter4_detection/tests -q
```

Use a visible CUDA device appropriate to the host. The formal run used device 1; device 0 above is a portable default.
