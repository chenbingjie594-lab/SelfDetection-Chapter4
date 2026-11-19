# Geometric-supervision failure analysis

## Same-GT paired comparison

For 1,076 augmented GTs containing both types, metrics were averaged within GT/type and then paired. Confidence intervals are a fixed-seed, 10,000-resample GT bootstrap of the outside-minus-inside mean.

| Metric | Inside mean | Outside mean | Outside − inside (95% CI) |
|---|---:|---:|---:|
| Predicted IoU | 0.556 | 0.406 | −0.150 [−0.165, −0.135] |
| Target score | 0.492 | 0.140 | −0.352 [−0.371, −0.332] |
| Unweighted DFL loss | 0.754 | 0.736 | −0.019 [−0.049, 0.012] |
| Weighted DFL contribution | 0.291 | 0.064 | −0.226 [−0.237, −0.215] |
| Weighted CIoU contribution | 0.151 | 0.046 | −0.106 [−0.111, −0.100] |
| Center error (input px) | 2.031 | 3.000 | +0.969 [+0.781, +1.166] |
| Maximum representable IoU | 1.000 | 0.753 | −0.247 [−0.256, −0.237] |

External positives are lower-quality and less heavily weighted, even within the same augmented GT. The unweighted DFL loss is not detectably larger, consistent with endpoint targets being learnable. This comparison is descriptive: TAL selects anchors using prediction-dependent alignment, so it does not isolate a causal effect of geometry.

## Interpretation and GVDS gate

1. External positives exist: **pass**.
2. They contain non-representable signed edges: **pass**.
3. They carry non-negligible total weight (37.75%): **pass**.
4. Evidence that endpoint DFL supervision itself is harmful: **not passed**.

The fourth gate fails because the clipped endpoint is the constrained geometric optimum, unweighted DFL loss is not worse, and low-IoU-max positives receive only 0.42% (<.50) or 5.92% (<.75) of total weight. Masking those edges would remove valid boundary supervision without evidence that it improves shared regression features. Consequently GVDS was not implemented or trained.

Training geometry may explain part of tiny-object localization weakness: tiny positives are outside more often and tiny GTs more often have only outside positives. It does not by itself establish causation for Stage12 OOF failures. Fold0 augmented training GTs cannot be safely one-to-one matched to cross-fold OOF instances, so no forced correlation was reported.

The 25 annotation-review candidates, including `img_1013_pre_part_1.jpg` and `img_9509_pre_part_0.jpg`, remain `PENDING_EXPERT_REVIEW`; no GT was changed or excluded.
