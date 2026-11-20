# Stage18 reproduction commands

```bash
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
PYTHONPATH=. $PY -m pytest -q chapter4_detection/tests/test_rglcd.py
PYTHONPATH=. $PY chapter4_detection/scripts/stage18_build_rglcd_init.py --baseline chapter4_detection/results/stage7/initializations/matched_e0_init_seed42.pt --output chapter4_detection/results/stage18/initializations/rglcd_matched_e0_seed42.pt --seed 42
PYTHONPATH=. $PY chapter4_detection/scripts/stage18_rglcd_audit.py --baseline chapter4_detection/results/stage7/initializations/matched_e0_init_seed42.pt --image /path/to/fold0/train/image.jpg --label /path/to/fold0/train/label.txt --output chapter4_detection/results/stage18/rglcd_gradient_audit.json --imgsz 640
PYTHONPATH=. $PY chapter4_detection/scripts/stage18_mechanism_diagnosis.py --weights chapter4_detection/results/stage18/initializations/rglcd_matched_e0_seed42.pt --train-list chapter4_detection/results/stage10/cv_protocol/fold0/train.txt --output chapter4_detection/results/stage18 --imgsz 640
```

`stage18_train.py` is retained as the frozen formal entry point, but Stage18's failed pre-training gate prohibits launching it.

