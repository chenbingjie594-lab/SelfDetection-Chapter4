# Signed scale residual learnability

## Protocol

Scope is the 92 Fold0-training images only. Candidates are generated independently of GT by the frozen Stage12 E0 rule (`reason=kept`); class-aware greedy IoU≥0.30 matching is used afterward only to construct supervision. Of 115 GT, 112 have a matched candidate, 3 do not, and 404 kept candidates are unmatched background. A scale module cannot recover the unmatched GT.

Five source-group folds separately fit every scaler, model and training-fold median. Inputs are prediction geometry, level, score and low-dimensional DFL statistics; no GT geometry, oracle IoU, candidate rank or repair result is an input. This uses frozen OOF feature caches but remains a small internal mechanism probe, not detection-model generalization evidence.

| method | width MAE | height MAE | combined MAE | IoU before→after | coverage .75 before→after |
|---|---:|---:|---:|---:|---:|
| zero | 0.1176 | 0.1245 | 0.1210 | 0.7389→0.7389 | 0.4732→0.4732 |
| fold median | 0.1151 | 0.1251 | 0.1201 | 0.7389→0.7413 | 0.4732→0.4821 |
| linear | 0.1153 | 0.1183 | 0.1168 | 0.7389→0.7363 | 0.4732→0.4732 |
| ridge | 0.1116 | 0.1178 | 0.1147 | 0.7389→0.7387 | 0.4732→0.4554 |
| HistGBR | 0.1073 | 0.1242 | 0.1158 | 0.7389→0.7410 | 0.4732→0.4554 |

Ridge is the best fixed probe by combined MAE, improving only 4.52% over the fold-local median, below the frozen 10% threshold. It improves 4/5 fold MAEs and both axes, but mean IoU changes by -0.0002, IoU≥.75 coverage changes by -0.0179, and 8 initially accurate candidates fall below 0.75. Bootstrap median-minus-ridge MAE improvement CI95 is [-0.0012, 0.0122], crossing zero.

Flash and Black do not justify separate mechanisms: ridge MAE is 0.1179 versus 0.1118, with mixed IoU effects. Tiny IoU rises slightly but high-quality coverage is unchanged. Decision: `NO_STABLE_SIGNED_RESIDUAL_SIGNAL`.
