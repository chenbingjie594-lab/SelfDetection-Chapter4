# Stage20 复现命令

在仓库根目录运行：

```bash
export CH4_PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python

$CH4_PY chapter4_detection/scripts/stage20_geometry.py \
  --candidates chapter4_detection/results/stage12/candidate_trace.csv \
  --gt-stats chapter4_detection/results/stage10/gt_statistics.csv \
  --failures chapter4_detection/results/stage12/gt_failure_decomposition.csv \
  --output chapter4_detection/results/stage20

PYTHONPATH=. $CH4_PY chapter4_detection/scripts/stage20_p3_boundary_analysis.py \
  --weights chapter4_detection/results/stage10/runs/e0_fold0_seed42/weights/last.pt \
  --train-list chapter4_detection/results/stage10/cv_protocol/fold0/train.txt \
  --gt-stats chapter4_detection/results/stage10/gt_statistics.csv \
  --output chapter4_detection/results/stage20 --imgsz 1536

PYTHONPATH=. $CH4_PY -m pytest -q chapter4_detection/tests/test_stage20_geometry.py
nvidia-smi
$CH4_PY -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); print(torch.ones(1, device='cuda'))"
```

由于机制门槛失败，没有 P3-BAR 正式训练命令；CUDA 检查失败时也不得用 CPU 启动三折 150 epoch 训练。
