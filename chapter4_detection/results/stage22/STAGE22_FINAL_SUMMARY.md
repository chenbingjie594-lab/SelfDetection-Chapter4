# Stage22 final summary

The trainable CIR prototype and all 12 CPU runs completed.

| Model | P | R | mAP50 | mAP50-95 | AP75 | Flash AP | Black AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0.8408 | 0.8047 | 0.8061 | 0.3540 | 0.2442 | 0.2556 | 0.4523 |
| A | 0.8696 | 0.8424 | 0.8528 | 0.3662 | 0.2371 | 0.2985 | 0.4340 |
| B | 0.8755 | 0.8477 | 0.8672 | 0.3720 | 0.2315 | 0.3060 | 0.4380 |
| C | 0.8755 | 0.8477 | 0.8610 | 0.3743 | 0.2506 | 0.3047 | 0.4438 |
| C-noDiff | 0.8692 | 0.8416 | 0.8511 | 0.3644 | 0.2349 | 0.2851 | 0.4437 |

CIR retention decision: **STOP**. Original-pixel, fusion, explicit-difference, E0 non-inferiority, and candidate-coverage conditions must all hold; the frozen joint rule failed. Therefore CIR does not remain a formal Chapter-4 innovation candidate. This is development OOF evidence only.
