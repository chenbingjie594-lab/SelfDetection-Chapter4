# Stage25 final summary

| Model | P | R | mAP50 | mAP50-95 | AP75 | Flash AP | Black AP | Low Black R | Tiny R | BG FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0.8408 | 0.8047 | 0.8061 | 0.3540 | 0.2442 | 0.2556 | 0.4523 | 0.6957 | 0.8077 | 6 |
| M2 | 0.8005 | 0.8413 | 0.8179 | 0.3593 | 0.2419 | 0.2653 | 0.4532 | 0.8261 | 0.8462 | 24 |
| M3 | 0.8148 | 0.8184 | 0.8164 | 0.3624 | 0.2387 | 0.2636 | 0.4612 | 0.8261 | 0.8462 | 30 |
| H1 | 0.7723 | 0.7790 | 0.7468 | 0.3236 | 0.1940 | 0.2227 | 0.4245 | 0.7826 | 0.7885 | 60 |
| H2 | 0.7805 | 0.7932 | 0.7957 | 0.3523 | 0.2337 | 0.2548 | 0.4498 | 0.6957 | 0.7885 | 39 |
| H3 | 0.7768 | 0.7856 | 0.7836 | 0.3491 | 0.2372 | 0.2465 | 0.4517 | 0.7391 | 0.8077 | 35 |
| H4 | 0.8237 | 0.7665 | 0.7672 | 0.3356 | 0.2075 | 0.2365 | 0.4348 | 0.7826 | 0.8077 | 80 |

## Status

This is a **weak-supervision feasibility exploration**, not a confirmed hard-background validation: A=0 and only 12–15 B candidates/fold were available. **STOP: retain none of H1–H4.** H2 is the least damaging new ablation by aggregate mAP50-95, but it remains below historical M2/M3 and does not solve background FP. H3 is not a formal innovation candidate and must not be combined with CIR.

## Required answers

1. Stage24's legacy FP mixed true background and class confusion. Under the uniform taxonomy E0/M2/M3 have respectively 6/24/30 background-style predictions and 4/4/4 class confusions.
2. Stage24 replicate-padding patches were not equivalent at P3 horizontal boundaries (maximum discrepancy 6.29). Stage25's corrected fresh-patch/full-map error is about 1e-5 and FP16 mean error is below 4e-4.
3. No. A=0; folds contain only 15/12/15 B candidates and 3/1/1 excluded C candidates.
4. No. H1 is below historical M3: mAP50-95 0.3236 versus 0.3624, with 60 versus 30 background FP.
5. H3 beats H1 by +0.0255 mAP50-95, but still fails against E0/M2/M3.
6. H3 beats H4 by +0.0135, but both are poor and use unconfirmed weak negatives.
7. No. H3 is below H2 by -0.0032; its paired CI crosses zero.
8. No. Texture matching has no independent contribution: H3 does not beat H2 and is -0.0133 below historical M3.
9. No robust reduction. H3 Flash AP is 0.2465 versus M3 0.2636 and E0 0.2556.
10. H3 low-contrast Black recall is 0.7391, above E0 0.6957 but below M3 0.8261.
11. No. The frozen-grid fixed-FPPI comparison and paired interval are in `HBD_RECALL_FPPI_ANALYSIS.md` and `HBD_BOOTSTRAP.json`.
12. H3 versus E0 changes mAP50/mAP50-95/AP75 by -0.0225/-0.0048/-0.0069; versus M3 by -0.0327/-0.0133/-0.0015.
13. The tiny cached B set is easily separated, but holdout FROC/PR worsens; this is not useful discrimination or calibration.
14. Yes. All regression and P4/P5 outputs are frozen and audited exactly.
15. No. A=0, fold instability and inferior fixed-FPPI recall prevent a mechanism claim.
16. Retain none of H1–H4. Keep E0 as formal baseline and M2/M3 only as historical controls; CIR remains the first innovation candidate.
17. No. HBD should not be combined with CIR.
