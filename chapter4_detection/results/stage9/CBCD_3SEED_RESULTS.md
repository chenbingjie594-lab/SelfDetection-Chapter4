# CBCD Three-Seed Results

All formal runs are complete. Values are mean ± sample SD.

| Model | Precision | Recall | mAP50 | mAP50-95 | AP75 | Flash AP | Black AP | Black Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Matched-E0 | 0.8431 ± 0.0391 | 0.7715 ± 0.0194 | 0.8065 ± 0.0243 | 0.4211 ± 0.0239 | 0.4140 ± 0.0648 | 0.4279 | 0.4142 | 0.6354 |
| SFDFM | 0.8516 ± 0.0127 | 0.8168 ± 0.0357 | 0.8436 ± 0.0157 | 0.4203 ± 0.0167 | 0.3391 ± 0.0190 | 0.4352 | 0.4055 | 0.6975 |
| CBCD | 0.8081 ± 0.0471 | 0.7651 ± 0.0418 | 0.8014 ± 0.0343 | 0.4367 ± 0.0082 | 0.4002 ± 0.0338 | 0.4576 | 0.4158 | 0.6418 |

| Seed | mAP50 | mAP50-95 | AP75 | Black AP | Black Recall | ΔmAP50-95 vs E0 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.8389 | 0.4453 | 0.4352 | 0.4185 | 0.6434 | -0.0032 |
| 3407 | 0.7715 | 0.4291 | 0.3678 | 0.4065 | 0.5824 | +0.0207 |
| 2026 | 0.7937 | 0.4356 | 0.3976 | 0.4223 | 0.6996 | +0.0294 |

CBCD improved the three-seed mean mAP50-95 by 0.0156, but the improvement was not present in seed42. Mean AP75 decreased by 0.0138; seed deltas were -0.0533, -0.0040 and +0.0160. It avoided SFDFM's large localization collapse, but did not establish a stable localization advantage.

Source-group bootstrap (5,000 replicates) estimated mean ΔmAP50-95 +0.0136, 95% CI [-0.0210, 0.0444], P(Δ>0)=0.800. ΔAP75 was -0.0108, CI [-0.0850, 0.0614]. Both intervals cross zero.

Low-contrast Black detections were 3/4/4 for CBCD and 4/4/3 for E0; small Black detections were 5/4/4 versus 5/3/4. Neither is a repeatable three-seed improvement. Fixed-threshold background-error counts were also seed-dependent.

CBCD has 9,447,254 unfused parameters; the fused checkpoint reports 9,432,262 parameters and 21.7 GFLOPs. Training peak allocated memory was 1.81/1.90/1.90 GB. Checkpoint SHA256: seed42 `5ffe7619...2bf0`, seed3407 `def6c400...f80b`, seed2026 `58580cc4...7920`.
