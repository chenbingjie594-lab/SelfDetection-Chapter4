# FSPR training results

All nine CPU cached-feature runs completed 40 epochs with non-zero required gradients for applicable branches and no holdout early stopping. F0 intentionally has zero P2 gradient because its P2 inputs are the frozen zero ablation.

- Fold 0 F0: loss=0.2478, IoU=0.2141, offset=0.0093, edge=0.0145.
- Fold 0 F1: loss=0.1917, IoU=0.1698, offset=0.0056, edge=0.0103.
- Fold 0 F2: loss=0.1757, IoU=0.1535, offset=0.0048, edge=0.0096.
- Fold 1 F0: loss=0.2285, IoU=0.1972, offset=0.0093, edge=0.0155.
- Fold 1 F1: loss=0.1746, IoU=0.1574, offset=0.0057, edge=0.0114.
- Fold 1 F2: loss=0.1640, IoU=0.1471, offset=0.0048, edge=0.0114.
- Fold 2 F0: loss=0.2804, IoU=0.2123, offset=0.0089, edge=0.0160.
- Fold 2 F1: loss=0.1937, IoU=0.1732, offset=0.0059, edge=0.0123.
- Fold 2 F2: loss=0.1778, IoU=0.1594, offset=0.0054, edge=0.0099.
