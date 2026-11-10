# Plastic_Bomo 第四章第一阶段：高分辨率微小目标检测

本目录实现三个预注册对照：E0（YOLO11s 整图缩放）、E1（YOLO11s + P2）和
E2（YOLO11s 原始尺度切片）。只允许冻结的 Real-only 图像进入训练。

## 当前门禁状态

**BLOCKED，禁止正式训练。** 候选数据确认为：

`/mnt/sda/cbj/dataset/Plastic_Bomo/DWBG_v2_final_s42/real_only.yaml`

它满足 138 张 train、46 张 val、训练 168 个实例（Flash 88、Black 80）、全部
4096×1024、YOLO HBB 两类，并与历史审计的 Real-only 数据绑定一致。然而保守的
canonical source ID 审计发现两组显式同源 `pre_part` sibling 跨越 train/val：

- `img_1020_pre_part_0.jpg`（train）↔ `img_1020_pre_part_1.jpg`（val）
- `img_9509_pre_part_0.jpg`（train）↔ `img_9509_pre_part_1.jpg`（val）

历史 Poisson 泄漏修复报告也记录了 `source group overlap = 2`。虽然没有字节级
SHA256 重复，但按本阶段规则，这仍属于已知亲缘关系风险。必须先由数据负责人确认
该关系是否构成任务定义中的泄漏；本仓库不会自行重新划分，也不会替换成 136/48。

详细证据见 `results/data_audit.json` 和 `results/DATA_AUDIT_REPORT.md`。

## 环境

- Python 3.10.20（`/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python`）
- Ultralytics 8.4.145
- PyTorch 2.5.1+cu124
- 当前执行会话：CUDA/NVML 不可用，因此未运行 GPU smoke 或正式训练
- 预训练权重 SHA256：`85a76fe...02d502d5`，与第三章历史 `pretrained/yolo11s.pt` 完全一致

统一训练协议来自历史实际 `args.yaml`：`imgsz=1536, batch=1, epochs=150,
patience=40, optimizer=auto, lr0=0.01, mosaic=1.0, close_mosaic=10`。第四章将这些参数
固定在 `configs/common.yaml`；旧结果只作历史参考，不冒充第四章新实验。

## 结构

- `configs/`：统一协议及 E0/E1/E2 差异
- `models/yolo11s-p2.yaml`：P2/P3/P4/P5 检测头
- `scripts/audit_data.py`：数量、标签、尺寸、SHA256、canonical/source group 和合成混入门禁
- `scripts/prepare_slices.py`：E2 训练切片、坐标裁剪及覆盖审计
- `scripts/infer_sliced.py`：固定切片、恢复原图坐标、全图 class-aware NMS
- `scripts/evaluate_original.py`：以 46 张原图为单位计算 P/R/AP
- `scripts/model_audit.py`：参数迁移、未匹配键、参数量、FLOPs 和 stride
- `scripts/train.py`：统一且强制数据门禁的训练入口
- `scripts/evaluate_model.py`：E0/E1 固定验证及扩展指标
- `scripts/freeze_diagnostic_thresholds.py`：仅用 train GT 冻结尺寸/对比度阈值
- `scripts/diagnose_predictions.py`：尺寸、对比度、IoU、分类漏检和背景误检诊断（非正式 AP）
- `tests/`：坐标变换、跨切片覆盖、NMS、E0/E1 前向和 P2 迁移测试

## 可直接运行的命令

所有命令从仓库根目录运行：

```bash
export PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
export PYTHONPATH=$PWD

# 环境检查
nvidia-smi
$PY -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__, torch.cuda.is_available())"

# 冻结数据审计；当前应以退出码 2 阻止训练
$PY -m chapter4_detection.scripts.audit_data \
  --data /mnt/sda/cbj/dataset/Plastic_Bomo/DWBG_v2_final_s42/real_only.yaml \
  --historical-audit /mnt/sda/cbj/ImageGeneration/Self-main/results_for_gpt/bootstrap_guard_downstream/dataset_audit.json \
  --output chapter4_detection/results/data_audit.json

# 自动化测试
$PY -m unittest discover -s chapter4_detection/tests -v

# 模型审计
$PY -m chapter4_detection.scripts.model_audit --model yolo11s.pt --imgsz 1536 \
  --output chapter4_detection/results/e0_model_audit.json
$PY -m chapter4_detection.scripts.model_audit \
  --model chapter4_detection/models/yolo11s-p2.yaml --weights yolo11s.pt --imgsz 1536 \
  --output chapter4_detection/results/e1_model_audit.json
```

以下命令只有在 `data_audit.json` 经审查变为 `PASS` 后才会运行；当前会失败退出：

```bash
# E0 seed42
$PY -m chapter4_detection.scripts.train --experiment E0 --seed 42 \
  --config chapter4_detection/configs/e0_yolo11s.yaml

# E1 seed42
$PY -m chapter4_detection.scripts.train --experiment E1 --seed 42 \
  --config chapter4_detection/configs/e1_yolo11s_p2.yaml

# E2 数据准备（只读取审核通过的 138 张 train）
$PY -m chapter4_detection.scripts.prepare_slices \
  --data /mnt/sda/cbj/dataset/Plastic_Bomo/DWBG_v2_final_s42/real_only.yaml \
  --audit-report chapter4_detection/results/data_audit.json \
  --output /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_e2_tiles_s42 \
  --tile-size 1024 --overlap 128 --min-visibility 0.5

# E2 seed42
$PY -m chapter4_detection.scripts.train --experiment E2 --seed 42 \
  --config chapter4_detection/configs/e2_yolo11s_slice.yaml \
  --data /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_e2_tiles_s42/data.yaml
```

正式评估：

```bash
# E0/E1
$PY -m chapter4_detection.scripts.evaluate_model --weights WEIGHTS.pt \
  --data /mnt/sda/cbj/dataset/Plastic_Bomo/DWBG_v2_final_s42/real_only.yaml \
  --imgsz 1536 --output RESULT.json

# E2 必须恢复到原图后评估，禁止用切片级 mAP
$PY -m chapter4_detection.scripts.infer_sliced --weights WEIGHTS.pt \
  --images /mnt/sda/cbj/dataset/Plastic_Bomo/DWBG_v2_final_s42/real_only/images/val \
  --tile-size 1024 --overlap 128 --output PREDICTIONS.json
$PY -m chapter4_detection.scripts.evaluate_original --predictions PREDICTIONS.json \
  --labels /mnt/sda/cbj/dataset/Plastic_Bomo/DWBG_v2_final_s42/real_only/labels/val \
  --output RESULT.json

# 诊断指标（与正式 AP 分开输出）
$PY -m chapter4_detection.scripts.diagnose_predictions --predictions PREDICTIONS.json \
  --images /mnt/sda/cbj/dataset/Plastic_Bomo/DWBG_v2_final_s42/real_only/images/val \
  --labels /mnt/sda/cbj/dataset/Plastic_Bomo/DWBG_v2_final_s42/real_only/labels/val \
  --thresholds chapter4_detection/results/diagnostic_thresholds.json \
  --output DIAGNOSTICS.json
```

数据门禁通过且 seed42 三组流程确认后，将训练命令中的 seed 依次改为 `3407`、
`2026`。未完成三个种子前不得报告 Mean ± SD。
