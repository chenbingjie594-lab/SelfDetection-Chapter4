# Stage14 final summary

## Answers to the required questions

1. **Outside original GT:** 4,125/6,625 positives, 62.26%.
2. **Weight share:** 37.75%.
3. **Negative DFL distances:** yes; 99.98% of outside positives have at least one.
4. **Clipping:** each signed target is clipped to `[0, 14.99]` bins before two-bin DFL interpolation.
5. **Representability:** among outside positives, 15.56% have IoU-max <.50 and 51.59% <.75. Across all positives, weight-adjusted shares are only 0.42% and 5.92%.
6. **Localization quality:** yes, descriptively. Same-GT outside positives have mean IoU 0.406 vs 0.556 and center error 3.000 vs 2.031 input pixels.
7. **Tiny defects:** geometry is a plausible partial contributor: 82.16% of tiny positives are outside and 35.92% of tiny augmented GT groups have all positives outside. Causality remains unproven.
8. **GVDS improvement:** not tested because the predeclared development gate failed; no result is fabricated.
9. **Innovation:** masking invalid DFL edges is a narrow loss-mask variant, not yet a substantiated innovation. The current endpoint supervision is geometrically defensible.
10. **Next stage:** GVDS is not recommended for multi-seed experiments. Prioritize candidate ranking/localization quality and annotation review; if revisiting assignment geometry, change anchor eligibility/parameterization under a separately frozen hypothesis rather than silently masking endpoint targets.

## Formal run

- Fold0/seed42 completed 150/150 epochs.
- Final `best.pt` validation: P=0.845, R=0.753, mAP50=0.751, mAP50-95=0.317.
- Final `last.pt` epoch validation recorded P=0.823, R=0.735, mAP50=0.724, mAP50-95=0.313.
- Runtime 0.321 h; reported GPU memory 1.88 GB.
- `last.pt` SHA256: `b6c093dbc8e3bc2c80d2dc7fa83117dc3b0c049a7f03d1dde75cfb3275892756`.

Metrics are mechanism-run integrity evidence, not a new model result: the run is exactly identical to Stage10 baseline weights and non-time training log.

## Decision

**Stop after diagnosis. Do not implement GVDS.** Three of four gates pass, but the necessary evidence that clipped endpoint DFL supervision is harmful does not. This avoids converting correlation into an unnecessary model modification.
