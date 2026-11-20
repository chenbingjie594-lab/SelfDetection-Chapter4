# Hard-background reference analysis

Sampling was fixed before interpretation: 92 Fold0-train images, seed42 deterministic G1 points in real content, four highest valid P3 `|D|` cells/image for G2, and four highest trained-E0 P3 classification responses/image for G3. GT plus three P3 cells and all letterbox padding were excluded. G3 entries are `HARD_BACKGROUND_CANDIDATES`, not confirmed false positives.

Trained-E0 at 1536:

| group | n | abs(D) | V | r | suppression | class response |
|---|---:|---:|---:|---:|---:|---:|
| G1 ordinary | 365 | .691 | .04691 | .96218 | 3.22% | ~0 |
| G2 high contrast | 368 | 4.423 | .03513 | .96988 | 3.02% | .0024 |
| G3 high class | 368 | 2.690 | .06330 | .94968 | 4.89% | .0059 |

G2/G3 are harder than ordinary background, and some visually resemble defects or strong texture. Twelve top G3 plates plus `HARD_BACKGROUND_REVIEW.csv` are supplied for human review. None is automatically labelled FP because annotations may be incomplete.

