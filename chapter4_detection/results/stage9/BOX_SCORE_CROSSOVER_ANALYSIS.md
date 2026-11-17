# Box–Score Crossover Analysis

All six model/seed exports used the exact validator preprocessing path. E0 and SFDFM had identical P3/P4/P5 shapes, strides, candidate counts, class order, input shape and ratio/pad for every image. C0/C1 reproduced the formal metrics exactly before C2/C3 were interpreted.

Three-seed means:

| Combination | Boxes | Scores | mAP50 | mAP50-95 | AP75 |
|---|---|---|---:|---:|---:|
| C0 | E0 | E0 | 0.8065 | 0.4211 | 0.4140 |
| C1 | SFDFM | SFDFM | 0.8436 | 0.4203 | 0.3391 |
| C2 | E0 | SFDFM | 0.8125 | 0.4226 | 0.4027 |
| C3 | SFDFM | E0 | 0.8270 | 0.4160 | 0.3649 |

Keeping E0 boxes and substituting SFDFM scores retained most AP75 and slightly increased mAP50/mAP50-95. Keeping SFDFM boxes and restoring E0 scores recovered only part of AP75. Therefore the large AP75 loss is primarily associated with box geometry; score ranking/NMS has a secondary, seed-dependent effect. This is an inference intervention, not an independent causal decomposition.
