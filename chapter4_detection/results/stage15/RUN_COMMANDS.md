# Stage15 reproduction commands

Run from the repository root in the existing YOLO11 environment. The scripts fail on unexpected image/input shape.

```bash
/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python -m chapter4_detection.scripts.stage15_resolution_audit \
  --dataset /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1 \
  --manifest /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/split_manifest.json \
  --output chapter4_detection/results/stage15/information_loss

/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python -m chapter4_detection.scripts.stage15_slice_audit \
  --dataset /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1 \
  --slices /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1_slices_1024_o128 \
  --output chapter4_detection/results/stage15/e2_audit

# Repeat fold=0,1,2 and condition=B1,B2.
/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python -m chapter4_detection.scripts.stage15_native_export \
  --weights chapter4_detection/results/stage10/runs/e0_fold0_seed42/weights/last.pt \
  --image-list chapter4_detection/results/stage10/cv_protocol/fold0/holdout.txt \
  --output chapter4_detection/results/stage15/multiscale/b1_fold0_predictions.json \
  --cache-dir chapter4_detection/results/stage15/raw_candidates/b1_fold0 --condition B1 --device 3
```

Subsequent analysis entry points are `stage15_analyze_multiscale`, `stage15_bootstrap_relation`, `stage15_benchmark`, `stage15_visualize_differences`, and `stage15_cache_manifest`.
