# SFDFM Three-Seed Results

Protocol: frozen real-only `chapter4_real_only_v1` (138 train/46 val, 44 validation source groups); matched Stage-7 E0 initialization; imgsz 1536, batch 1, up to 150 epochs, patience 40. Seed 3407 stopped at epoch 75; seeds 42/2026 completed 150. Both models use full-image validation mAP50-95 checkpoint selection.

Values are mean ± sample SD over three seeds.

| Model | Precision | Recall | mAP50 | mAP50-95 | AP75 | Flash AP50-95 | Black AP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Matched-E0 | 0.8431 ± 0.0391 | 0.7715 ± 0.0194 | 0.8065 ± 0.0243 | 0.4211 ± 0.0239 | 0.4140 ± 0.0648 | 0.4279 ± 0.0377 | 0.4142 ± 0.0101 |
| SFDFM | 0.8516 ± 0.0127 | 0.8168 ± 0.0357 | 0.8436 ± 0.0157 | 0.4203 ± 0.0167 | 0.3391 ± 0.0190 | 0.4352 ± 0.0406 | 0.4055 ± 0.0076 |

| Seed | SFDFM P | R | mAP50 | mAP50-95 | AP75 | ΔmAP50-95 vs E0 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.8583 | 0.8524 | 0.8561 | 0.4348 | 0.3180 | -0.0138 |
| 3407 | 0.8596 | 0.7810 | 0.8489 | 0.4021 | 0.3448 | -0.0063 |
| 2026 | 0.8370 | 0.8169 | 0.8259 | 0.4241 | 0.3547 | +0.0179 |

SFDFM increased mean Recall by 0.0453 and mAP50 by 0.0371, but did not improve mAP50-95 (change -0.0007). AP75 fell by 0.0749, indicating more coarse detections without better localization.

Flash Recall/AP50/AP50-95 were 0.9361/0.9152/0.4352 for SFDFM versus 0.9076/0.8915/0.4279 for E0. Black values were 0.6975/0.7720/0.4055 versus 0.6354/0.7215/0.4142. Thus Black coarse detection improved, while stricter averaged localization did not. Low-contrast and small-Black diagnostic recall did not improve consistently across seeds.

Source-group bootstrap (5,000 replicates) gave three-seed mean ΔmAP50-95 -0.00046, 95% CI [-0.03465, 0.03306], P(Δ>0)=0.485; ΔAP75 -0.06847, CI [-0.16139, 0.02354], P(Δ>0)=0.072. Flash ΔAP was +0.00700 and Black ΔAP -0.00791; both intervals crossed zero.

SFDFM has 9,547,990 parameters and 23.0 GFLOPs at 1536 (unfused audit). Training peak allocated GPU memory was 2.05/1.97/2.03 GB. Validator-reported inference was 8.1/10.4/9.8 ms per image (mean 9.43 ms); this is not a separately synchronized end-to-end benchmark. CUDA was unavailable when that follow-up benchmark was attempted, so no timing was fabricated.

Best checkpoint SHA256 prefixes: seed42 `02a65519`, seed3407 `e131d178`, seed2026 `e565f20e`. Checkpoints remain local and Git-ignored; metrics, predictions, diagnostics and logs are tracked.
