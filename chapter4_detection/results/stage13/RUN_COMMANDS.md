# 复现命令

```bash
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
$PY -m pytest chapter4_detection/tests/test_assignment_monitor.py -q
MPLCONFIGDIR=/tmp/mpl-stage13 $PY -m chapter4_detection.scripts.stage13_anchor_coverage --gt-stats chapter4_detection/results/stage10/gt_statistics.csv --manifest /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/split_manifest.json --output-dir chapter4_detection/results/stage13
$PY -m chapter4_detection.scripts.stage13_static_assignment --repo "$PWD" --output-dir chapter4_detection/results/stage13/static_assignment --device 1
$PY -m chapter4_detection.scripts.stage13_dynamic_train --data chapter4_detection/results/stage10/cv_protocol/fold0/data.yaml --init chapter4_detection/results/stage7/initializations/matched_e0_init_seed42.pt --project chapter4_detection/results/stage13/runs --device 1
MPLCONFIGDIR=/tmp/mpl-stage13 $PY -m chapter4_detection.scripts.stage13_summarize --stage13 chapter4_detection/results/stage13 --stage12 chapter4_detection/results/stage12
```

诊断 last.pt SHA256：`7a2073fc309922506be99ef54a57bc541edef35ae6f6f4cb777f8388cfb7ec2f`。它与 Stage10 Fold0 last.pt 的499个 state tensor逐元素相同，文件哈希因checkpoint封装元数据不同而不同。大权重仅本地保留；可审计CSV、args、results和metadata纳入Git。
