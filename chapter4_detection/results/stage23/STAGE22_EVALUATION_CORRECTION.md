# Stage22 evaluation correction

Stage22 fixed the original best-candidate index when measuring refined coverage. That metric remains useful as `fixed_candidate_iou_change`, but it is not GT coverage after refinement. This correction independently re-selects the maximum-IoU same-class post-NMS candidate for every GT. It is an offline oracle diagnostic and is never used at inference.

| Model | Coverage@.75 | Mean best IoU | Tiny coverage | Gains | Losses | ID switches |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 80/174 | 0.7184 | 17/52 | 0 | 0 | 0 |
| A | 80/174 | 0.7217 | 18/52 | 16 | 16 | 17 |
| B | 82/174 | 0.7256 | 17/52 | 15 | 13 | 15 |
| C | 82/174 | 0.7258 | 17/52 | 15 | 13 | 15 |
| C-noDiff | 81/174 | 0.7220 | 18/52 | 15 | 14 | 16 |

The original Stage22 STOP decision is preserved. This correction does not retroactively alter its frozen joint decision rule or reports.
