# Hard-background mining audit

Candidates are mined strictly from each outer-fold training list and the fold-specific E0 decoded pre-NMS output. Positions inside twice-expanded GT regions, padding and TAL positives are excluded, and every retained decoded box has maximum IoU below 0.10 to every annotated GT. Scores in `[0.05,0.25)` are **B: unconfirmed conservative candidates** and receive weak weight 0.25; scores `>=0.25` are **C: possible missed defect/ambiguous** and are excluded. The CSV records restored original boxes, locations, E0 classes/scores, maximum GT IoU and nearest-GT center distance. No independent human-review record was available, so **A=0** and no candidate is represented as confirmed normal background. Review plates show the native-resolution crop (left) and corresponding 1536-square YOLO-input crop (right).

- Fold 0: A=0, B=15, C=3.
- Fold 1: A=0, B=12, C=1.
- Fold 2: A=0, B=15, C=1.
