# Plastic_Bomo 第四章第一阶段：真实数据检测基线

本目录实现 E0（YOLO11s 整图缩放）、E1（YOLO11s + P2）和 E2（YOLO11s
原始尺度切片）。协议只允许真实图像，不修改损失函数，也不加入其他模块。

## 数据协议

正式协议位于 `/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/`，状态为
**PASS**。184 张真实图像全部保留且无生成图：train 138 张/174 个实例
（Flash 81、Black 93），val 46 张/58 个实例（Flash 27、Black 31）。协议哈希：
`48054c137c8e093957c57eef6d761750cfac6f7b437e089cebe42053fb4b62ab`。

149 张 `img_*` 用原始 XML 的 `filename/path` 建立 source group；已确认并隔离
`img_1020_pre_part_0/1` 与 `img_9509_pre_part_0/1`。35 张 `blackinst_*` 使用历史
确定性导出 ID，其中 25 张有精确重建证据，10 张在 manifest 中保留较低证据等级。
审计同时检查文件/标签/解码像素 SHA256、HBB、尺寸、合成混入和近重复候选。
冻结文件为 `data.yaml`、`split_manifest.json`、`data_audit.json`、`DATA_PROTOCOL.md`。
旧 138/46 指标只能作历史参考，不能用于计算新协议增益。

## 三组协议与 checkpoint 规则

- E0：`imgsz=1536`，完整原图验证集 mAP50-95 选择 `best.pt`。
- E1：stride 4/8/16/32 的 P2/P3/P4/P5 head，完整原图验证集选择 `best.pt`。
- E2：1024×1024、overlap 128 原尺度训练切片；训练时 `val=False`、`patience=0`，
  正式 checkpoint 固定为第 150 轮 `last.pt`，不使用整图缩放验证选择模型。评估流程为
  完整原图固定切片→恢复坐标→全图 class-aware NMS→原图 GT 指标。

E1 显式迁移未改变的 backbone/top-down neck、语义相同的 bottom-up P3/P4/P5
层及检测回归分支。新增 P2 路径和语义/形状不兼容的分类头保留默认初始化。迁移模型
保存并重载为 `e1_semantic_initialized.pt`，防止训练器重建 YAML 后丢失权重。

## 环境与验证

- Python 3.10.20；Ultralytics 8.4.145；PyTorch 2.5.1+cu124。
- 13 项测试通过：source-group 隔离、已知 sibling、坐标转换、跨切片目标、坐标恢复、
  class-aware NMS、E0/E1 前向、显式迁移及 checkpoint 重载。
- E2 共 690 个训练切片；174/174 个目标均至少在一个切片中完整覆盖。
- GPU 后续已恢复。E0 seed42 已完成（第 76 轮早停，最佳第 36 轮）；其固定评估结果见
  `results/e0_seed42_metrics.json` 和 `results/E0_SEED42_REPORT.md`。E1、E2 与其余种子待运行。

## 可直接运行的命令

从仓库根目录执行：

```bash
export CH4_PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
export PYTHONPATH=$PWD
nvidia-smi
$CH4_PY -c "import torch,ultralytics; print(torch.__version__,ultralytics.__version__,torch.cuda.is_available())"

# 仅用于创建全新协议；目标存在时会拒绝覆盖
$CH4_PY -m chapter4_detection.scripts.build_data_protocol \
  --source-yaml /mnt/sda/cbj/dataset/Plastic_Bomo/DWBG_v2_final_s42/real_only.yaml \
  --xml-root /mnt/sda/cbj/dataset/PlasticBomo_detect2/metadata/original_xml \
  --black-evidence /mnt/sda/cbj/ImageGeneration/Self-main/results_for_gpt/plastic_bomo_stage20a_v2_protocol/evidence/black_exports.json \
  --output /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1 --seed 42

$CH4_PY -m unittest discover -s chapter4_detection/tests -v
$CH4_PY -m chapter4_detection.scripts.model_audit \
  --model chapter4_detection/models/yolo11s-p2.yaml --weights yolo11s.pt --imgsz 1536 \
  --output chapter4_detection/results/e1_model_audit_v2.json

$CH4_PY -m chapter4_detection.scripts.prepare_slices \
  --data /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/data.yaml \
  --audit-report /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/data_audit.json \
  --output /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1_slices_1024_o128 \
  --tile-size 1024 --overlap 128 --min-visibility 0.5
```

正式训练先逐项运行 seed 42 并检查，再运行另外两个种子：

```bash
for CH4_SEED in 42 3407 2026; do
  $CH4_PY -m chapter4_detection.scripts.train --experiment E0 --seed $CH4_SEED \
    --config chapter4_detection/configs/e0_yolo11s.yaml
  $CH4_PY -m chapter4_detection.scripts.train --experiment E1 --seed $CH4_SEED \
    --config chapter4_detection/configs/e1_yolo11s_p2.yaml
  $CH4_PY -m chapter4_detection.scripts.train --experiment E2 --seed $CH4_SEED \
    --config chapter4_detection/configs/e2_yolo11s_slice.yaml \
    --data /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1_slices_1024_o128/data.yaml
done
```

评估命令：

```bash
# E0/E1：WEIGHTS 使用各自 best.pt
$CH4_PY -m chapter4_detection.scripts.evaluate_model --weights WEIGHTS.pt \
  --data /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/data.yaml \
  --imgsz 1536 --output RESULT.json

# E2：WEIGHTS 必须使用固定最终轮 last.pt
$CH4_PY -m chapter4_detection.scripts.infer_sliced --weights WEIGHTS.pt \
  --images /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/images/val \
  --tile-size 1024 --overlap 128 --conf 0.001 --nms-iou 0.7 --output PREDICTIONS.json
$CH4_PY -m chapter4_detection.scripts.evaluate_original --predictions PREDICTIONS.json \
  --labels /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/labels/val \
  --output RESULT.json
$CH4_PY -m chapter4_detection.scripts.diagnose_predictions --predictions PREDICTIONS.json \
  --images /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/images/val \
  --labels /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/labels/val \
  --thresholds chapter4_detection/results/diagnostic_thresholds.json --output DIAGNOSTICS.json
```

未完成三个种子前只报告实际值；完成后才能汇总 Mean ± SD。
