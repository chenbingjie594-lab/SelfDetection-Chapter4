# CIR group-CV results

| Model | P | R | mAP50 | mAP50-95 | AP75 | Flash AP | Black AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0.8408 | 0.8047 | 0.8061 | 0.3540 | 0.2442 | 0.2556 | 0.4523 |
| A | 0.8696 | 0.8424 | 0.8528 | 0.3662 | 0.2371 | 0.2985 | 0.4340 |
| B | 0.8755 | 0.8477 | 0.8672 | 0.3720 | 0.2315 | 0.3060 | 0.4380 |
| C | 0.8755 | 0.8477 | 0.8610 | 0.3743 | 0.2506 | 0.3047 | 0.4438 |
| C-noDiff | 0.8692 | 0.8416 | 0.8511 | 0.3644 | 0.2349 | 0.2851 | 0.4437 |

## Per-fold direction

| Fold | Model | mAP50-95 | AP75 |
|---:|---|---:|---:|
| 0 | E0 | 0.3720 | 0.2884 |
| 0 | A | 0.3756 | 0.2855 |
| 0 | B | 0.3801 | 0.2956 |
| 0 | C | 0.3854 | 0.2971 |
| 0 | C-noDiff | 0.3720 | 0.2772 |
| 1 | E0 | 0.4182 | 0.2606 |
| 1 | A | 0.3849 | 0.2828 |
| 1 | B | 0.4021 | 0.2446 |
| 1 | C | 0.4047 | 0.3366 |
| 1 | C-noDiff | 0.3879 | 0.2960 |
| 2 | E0 | 0.3252 | 0.2118 |
| 2 | A | 0.3751 | 0.2001 |
| 2 | B | 0.3724 | 0.2078 |
| 2 | C | 0.3754 | 0.1988 |
| 2 | C-noDiff | 0.3699 | 0.2017 |

Effects are not directionally uniform across all folds. Source-group paired bootstrap intervals, including candidate and tiny-candidate coverage, are in `cir_group_bootstrap.json`. These are exploratory development OOF results, not independent-test generalization.
