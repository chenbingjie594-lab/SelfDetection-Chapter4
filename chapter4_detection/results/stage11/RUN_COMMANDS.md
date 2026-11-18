# 复现命令

在仓库根目录使用 `/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python`：

```bash
python -m chapter4_detection.scripts.stage11_dfl_diagnosis --help
python -m chapter4_detection.scripts.stage11_p2_geometry --help
python -m chapter4_detection.scripts.stage11_build_pcrf_init --help
bash chapter4_detection/scripts/run_stage11_pcrf_cv.sh
python -m pytest chapter4_detection/tests/test_pcrf.py -q
```

训练入口固定读取 Stage10 `fold_manifest.json`，每折写入独立目录并以 `last.pt` 评估。完整参数保存在各 run 的 `args.yaml` 与 `run_metadata.json`；大体积 raw logits、checkpoint 和训练目录按仓库策略不提交。
