# Information–localization relation

The GT-level B1−B2 Qmax change was joined to the frozen information-loss descriptors by image, GT ID, class, and source group.

| Information descriptor | Spearman with B1−B2 Qmax |
|---|---:|
| Gradient response-contrast loss | −0.036 |
| Laplacian response-contrast loss | +0.032 |
| Defect PSNR | +0.242 |
| Defect SSIM | +0.243 |

The predeclared gradient and Laplacian contrast-loss measures have essentially no relation to candidate improvement. The positive PSNR/SSIM correlations point in the opposite direction from a simple “greater reconstruction damage → greater native-detail benefit” hypothesis: images reconstructed more faithfully tended to show slightly larger B1 gains. This is descriptive and not independent confirmatory evidence.

The observed pattern is closest to Stage15 case D/C: B1 and B2 are both strongly changed relative to B0 and nearly equal to each other. Input-scale/feature-grid distribution shift dominates; retained original detail does not rescue the frozen model. The native model sees defects at about 2.67× the trained linear scale and far more background grid cells, changing receptive-field context and score distributions.

No GT was altered. The 25 annotation candidates, including `img_1013_pre_part_1.jpg` and `img_9509_pre_part_0.jpg`, remain `PENDING_EXPERT_REVIEW`.
