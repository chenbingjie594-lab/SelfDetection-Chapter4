# Stage-7 commands

Run from the repository root with the existing YOLO11 environment.

```bash
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
IMG=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/images/train/img_3122_pre_part_1.jpg
LBL=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/labels/train/img_3122_pre_part_1.txt
for seed in 42 3407 2026; do
  PYTHONPATH=. $PY chapter4_detection/scripts/stage7_initialization_audit.py --pretrained yolo11s.pt --image $IMG --label $LBL --output-dir chapter4_detection/results/stage7/initializations --seed $seed --imgsz 640 --device cuda:0
done
PYTHONPATH=. $PY chapter4_detection/scripts/gamma_gradient_analysis.py --help
PYTHONPATH=. $PY chapter4_detection/scripts/diagnose_training_dynamics.py --help
DEVICE=0 bash chapter4_detection/scripts/run_stage7_matched_experiments.sh
PYTHONPATH=. $PY chapter4_detection/scripts/feature_intervention_analysis.py --weights-dir chapter4_detection/results/runs --images /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/images/val --labels /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/labels/val --output-dir chapter4_detection/results/stage7 --device cuda:0
PYTHONPATH=. $PY chapter4_detection/scripts/analyze_final_intervention_predictions.py --input-dir chapter4_detection/results/stage6/interventions --output-dir chapter4_detection/results/stage7 --confidence 0.25
PYTHONPATH=. $PY chapter4_detection/scripts/summarize_stage7.py --stage7 chapter4_detection/results/stage7 --repo .
```

The two `resume_stage7_*.sh` scripts document the actual recovery commands used after process interruption; completed checkpoints and metrics were not overwritten.
