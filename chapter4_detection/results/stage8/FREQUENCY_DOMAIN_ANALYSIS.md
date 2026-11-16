# Frequency-Domain Analysis

## Protocol

Only the frozen training split was used. Each HBB produced a defect **candidate ROI**, a complete surrounding ring and a same-image, same-size, non-overlapping background ROI selected without validation information. Three incomplete boundary cases were excluded; 171 paired ROIs remained (Flash 81, Black 90). Intervals use 2,000 source-group bootstrap resamples.

## Main paired differences (candidate ROI − matched background)

| Metric | Flash mean [95% CI] | Black mean [95% CI] |
|---|---:|---:|
| Haar high ratio | 0.00128 [0.00103, 0.00155] | 0.00220 [0.00191, 0.00250] |
| FFT broad high ratio | -0.2229 [-0.2502, -0.1952] | -0.2176 [-0.2664, -0.1777] |
| Sobel mean | 0.1246 [0.1092, 0.1421] | 0.1735 [0.1556, 0.1925] |
| Laplacian absolute mean | 0.0220 [0.0167, 0.0272] | 0.0463 [0.0399, 0.0531] |
| Texture entropy | 1.1740 [1.0503, 1.2996] | 1.2552 [1.1278, 1.3884] |

Flash and Black candidate boxes both have stronger local directional/edge responses than matched background, and Black has the larger edge difference. However, broad FFT high-frequency proportion is *lower* in defect boxes: normal texture contains substantial broadband/periodic high-frequency energy. Therefore indiscriminate high-frequency amplification is contradicted by the data and could increase false positives. The supported mechanism is selective directional detail conditioned on spatial/cross-scale evidence.

HBBs are not segmentation masks; these results describe candidate regions and cannot prove that every high-response pixel is defective. Full statistics are in `frequency_roi_statistics.csv` and `frequency_analysis.json`.
