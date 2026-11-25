# Patch/full-map equivalence audit

The initial Stage24-style replicate-padded audit exposed boundary logit errors up to 6.29. Zero-padding the cached input alone was insufficient because adapter/classifier BatchNorm can make artificial padded positions non-zero between spatial layers. Stage25 therefore masks invalid positions after the adapter and after each spatial classification block.

| Fold | Positions | Fresh FP32 patch vs full map max | FP16 cache vs full map max | FP16 mean absolute error |
|---:|---:|---:|---:|---:|
| 0 | 192 | 1.14e-5 | 0.00272 | 0.000313 |
| 1 | 192 | 1.05e-5 | 0.00473 | 0.000363 |
| 2 | 192 | 9.54e-6 | 0.00164 | 0.000374 |

All BatchNorm layers are in evaluation mode. The remaining cached error is attributable to FP16 feature storage and is small relative to the original boundary error. Exact rows are in `audits/patch_equivalence_fold*.json`.
