# Paired Source-group Bootstrap Report

Protocol: 5,000 paired resamples of 44 source groups (46 images), seed 20261115. AP is recomputed from duplicated per-image predictions; duplicates retain independent statistical copies. No replicate lacked either class.

| Difference (seed mean) | Metric | Mean | 95% percentile CI | P(>0) |
|---|---|---:|---:|---:|
| E3-E0 | map50_95 | 0.0082 | [-0.0208, 0.0355] | 0.720 |
| E3-E0 | ap75 | 0.0349 | [-0.0563, 0.1262] | 0.767 |
| E3-E0 | flash_ap50_95 | 0.0100 | [-0.0182, 0.0384] | 0.758 |
| E3-E0 | black_ap50_95 | 0.0063 | [-0.0441, 0.0539] | 0.603 |
| E3-E1 | map50_95 | 0.0024 | [-0.0309, 0.0353] | 0.560 |
| E3-E1 | ap75 | 0.0271 | [-0.0571, 0.1099] | 0.741 |
| E3-E1 | flash_ap50_95 | -0.0176 | [-0.0662, 0.0335] | 0.247 |
| E3-E1 | black_ap50_95 | 0.0224 | [-0.0219, 0.0639] | 0.844 |
| E3-A1 | map50_95 | 0.0061 | [-0.0225, 0.0355] | 0.665 |
| E3-A1 | ap75 | -0.0145 | [-0.0995, 0.0702] | 0.368 |
| E3-A1 | flash_ap50_95 | -0.0187 | [-0.0607, 0.0302] | 0.210 |
| E3-A1 | black_ap50_95 | 0.0310 | [-0.0077, 0.0690] | 0.944 |
| E3-A2 | map50_95 | 0.0193 | [-0.0091, 0.0467] | 0.907 |
| E3-A2 | ap75 | 0.0285 | [-0.0428, 0.0979] | 0.782 |
| E3-A2 | flash_ap50_95 | 0.0041 | [-0.0344, 0.0413] | 0.591 |
| E3-A2 | black_ap50_95 | 0.0346 | [-0.0077, 0.0762] | 0.943 |
| A1-E0 | map50_95 | 0.0020 | [-0.0244, 0.0273] | 0.568 |
| A1-E0 | ap75 | 0.0494 | [-0.0292, 0.1235] | 0.907 |
| A1-E0 | flash_ap50_95 | 0.0287 | [-0.0140, 0.0684] | 0.914 |
| A1-E0 | black_ap50_95 | -0.0247 | [-0.0548, 0.0034] | 0.043 |

E3−E0 mAP50-95 is +0.0082, but its interval crosses zero. E3−E0 AP75 is +0.0349 with a wide interval crossing zero. A1−E3 AP75 is +0.0145 (the table expresses E3−A1 = −0.0145), also with a wide interval crossing zero. Thus neither localization claim is supported as dataset-sampling robust.

These intervals condition on frozen checkpoints and the current validation population. They neither replace an independent test nor fully model retraining randomness.
