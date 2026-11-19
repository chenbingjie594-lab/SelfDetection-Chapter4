# Multiscale OOF evaluation

## Detection results

| Experiment | mAP50 | mAP50-95 | AP75 | Flash AP50-95 | Black AP50-95 |
|---|---:|---:|---:|---:|---:|
| B0 original 1536 | 0.8061 | 0.3540 | 0.2442 | 0.2556 | 0.4523 |
| B1 native 4096×1024 | 0.0656 | 0.0150 | 0.0041 | 0.0054 | 0.0246 |
| B2 down/up 4096×1024 | 0.0713 | 0.0154 | 0.0033 | 0.0078 | 0.0230 |

B0 reproduction was an execution gate: Fold0 reproduced all 228 stored predictions with maximum coordinate/score difference 0 and exactly reproduced mAP50/mAP50-95.

Source-group paired bootstrap (137 groups, 5,000 resamples) gives B1−B2 mAP50-95 `−0.00012`, 95% interval `[−0.00599, 0.00623]`; AP75 `+0.00120 [−0.00391, 0.00755]`. These intervals cover image/source-group sampling only, not training-seed uncertainty.

## Efficiency

Fixed Fold0 checkpoint, 20 images ×3 repeats, three warmups, CUDA synchronization:

| Condition | Actual tensor | Total time/image | Peak allocated memory | Approx. GFLOPs* |
|---|---|---:|---:|---:|
| B0 | 1×3×1536×1536 | 40.35 ms | 319 MiB | 125 |
| B1 | 1×3×1024×4096 | 49.91 ms | 733 MiB | 222 |
| B2 | 1×3×1024×4096 | 114.53 ms | 733 MiB | 222 |

`*` Area-scaled estimate from YOLO11s 21.7 GFLOPs at 640²; not a hardware measurement. B2 timing includes the explicit down/up resize (47.53 ms). B1 costs 1.24× B0 wall time and 2.30× allocated memory in this benchmark, despite much worse accuracy.
