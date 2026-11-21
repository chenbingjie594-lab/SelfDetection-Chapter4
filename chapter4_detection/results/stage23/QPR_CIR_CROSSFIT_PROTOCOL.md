# QPR-CIR cross-fit protocol

Each Stage10 outer training split is partitioned into three inner source-group folds. Nine temporary CIR-C models use the frozen Stage22 40-epoch recipe; each creates quality labels only for its unseen inner fold. The merged labels train the outer gate. Outer holdout labels are never loaded by gate training. The outer E0 detector did see outer-training images, so this is not fully independent detector evidence.
