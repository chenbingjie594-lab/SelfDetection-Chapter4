# FSPR ablation results

R1 vs historical F1 tests natural sampling; R2 vs R1 tests N256 negatives; R3 vs R2 tests phase information. Increasing negatives reduced raw candidate quality on fold 0, and F2 did not recover it.

| Model | P | R | mAP50 | mAP50-95 | AP75 | Flash AP | Black AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0.8450 | 0.7533 | 0.7859 | 0.3720 | 0.2884 | 0.2302 | 0.5137 |
| R1 independent | 0.4006 | 0.7494 | 0.5338 | 0.2153 | 0.1177 | 0.1572 | 0.2734 |
| R1 raw fusion | 0.0574 | 0.6713 | 0.0536 | 0.0278 | 0.0238 | 0.0237 | 0.0318 |
| R1 calibrated@0.1 | 0.7917 | 0.7925 | 0.7724 | 0.3637 | 0.2778 | 0.2302 | 0.4971 |
| R2 independent | 0.3201 | 0.4751 | 0.3378 | 0.1367 | 0.0885 | 0.0792 | 0.1943 |
| R2 raw fusion | 0.2080 | 0.6557 | 0.1820 | 0.0994 | 0.0982 | 0.0229 | 0.1759 |
| R2 calibrated@0.1 | 0.6961 | 0.7930 | 0.7036 | 0.3272 | 0.2369 | 0.2103 | 0.4441 |
| R3 independent | 0.2638 | 0.4352 | 0.2679 | 0.1268 | 0.0773 | 0.0317 | 0.2220 |
| R3 raw fusion | 0.1774 | 0.6372 | 0.1703 | 0.0895 | 0.0781 | 0.0164 | 0.1625 |
| R3 calibrated@0.1 | 0.6396 | 0.7367 | 0.6390 | 0.2879 | 0.1933 | 0.2086 | 0.3673 |
