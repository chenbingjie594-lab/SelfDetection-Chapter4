# 复现命令

```bash
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
$PY -m pytest chapter4_detection/tests/test_stage12_traceable_nms.py -q
$PY -m chapter4_detection.scripts.stage12_candidate_analysis --raw-root chapter4_detection/results/stage11/raw --labels /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/labels/train --output-dir chapter4_detection/results/stage12
$PY -m chapter4_detection.scripts.stage12_materialize_interventions --trace chapter4_detection/results/stage12/candidate_trace.csv --nms-json chapter4_detection/results/stage12/nms_intervention_predictions.json --output-dir chapter4_detection/results/stage12/interventions
$PY -m chapter4_detection.scripts.stage12_quality_features --trace chapter4_detection/results/stage12/candidate_trace.csv --raw-root chapter4_detection/results/stage11/raw --output chapter4_detection/results/stage12/candidate_quality_features.csv
$PY -m chapter4_detection.scripts.stage12_summarize --stage12 chapter4_detection/results/stage12 --labels /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/labels/train --manifest /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/split_manifest.json
```

环境：Ultralytics 8.4.145；基准 commit `06112ddd3acae232e31856b6f9c2a926aa1d7270`。GPU驱动在本阶段不可通信，但全部分析复用已冻结的本地 raw OOF 缓存并在 CPU 完成，未重新推理或训练。缓存逐文件 SHA256 见 `analysis_summary.json`。
