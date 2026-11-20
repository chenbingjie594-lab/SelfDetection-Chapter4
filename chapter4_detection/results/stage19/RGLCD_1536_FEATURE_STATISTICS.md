# RGLCD 1536 feature statistics

All 92 Fold0 training images and 115 GT were used. No holdout label or metric was read. Images are 4096×1024; at 1536 the scale is .375 with a 1536×384 content band centered in a 1536 square, and P3 is 192×192. Background selection explicitly excludes letterbox padding and a three-cell margin around GT.

| state/scale/group | n | abs(D) | V | r | suppression |
|---|---:|---:|---:|---:|---:|
| Init 640 Flash | 54 | 4.015 | .01181 | .98842 | 1.31% |
| Init 640 Black | 55 | 3.392 | .01098 | .98929 | 1.16% |
| Init 1536 Flash | 54 | 4.487 | .01448 | .98579 | 1.59% |
| Init 1536 Black | 55 | 5.705 | .01273 | .98753 | 1.34% |
| Trained E0 1536 Flash | 54 | 9.769 | .05505 | .95400 | 4.06% |
| Trained E0 1536 Black | 55 | 8.796 | .07532 | .94190 | 5.68% |
| Trained E0 1536 low-contrast Black | 6 | 6.832 | .06813 | .94415 | 5.40% |

The 1536 scale materially changes feature statistics, especially after E0 training. The Stage18 “all tiny” error is removed: original dimensions, frozen 1536 input widths/heights, P3 center and P3 box coverage are retained per instance in `stage19_feature_instances.csv`. `phi(D)` and `r*phi(D)` are separate; gamma is not used to portray a trained enhancement.

