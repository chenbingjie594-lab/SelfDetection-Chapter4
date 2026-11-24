# Stage24 final summary

| Model | P | R | mAP50 | mAP50-95 | AP75 | Flash AP | Black AP | Low-contrast Black R | Tiny R | BG FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0.8408 | 0.8047 | 0.8061 | 0.3540 | 0.2442 | 0.2556 | 0.4523 | 0.6957 | 0.8077 | 10 |
| M1 | 0.7927 | 0.8345 | 0.8115 | 0.3516 | 0.2232 | 0.2605 | 0.4426 | 0.7826 | 0.8654 | 36 |
| M2 | 0.8005 | 0.8413 | 0.8179 | 0.3593 | 0.2419 | 0.2653 | 0.4532 | 0.8261 | 0.8462 | 28 |
| M3 | 0.8148 | 0.8184 | 0.8164 | 0.3624 | 0.2387 | 0.2636 | 0.4612 | 0.8261 | 0.8462 | 34 |

## Frozen decision

**STOP / NOT RETAIN**. M3 improves aggregate mAP50-95 over M2 by +0.0031, but its mAP50 (0.8164) and AP75 (0.2387) are below M2 (0.8179/0.2419); the source-group bootstrap mAP50-95 CI is [-0.005486890350641288, 0.011491954363133652]. Only 1/3 folds favour M3 over M2. Background FP rises from 10 to 34.

## Required mechanism answers

1. MBRD is integrated before the real P3 classification tower during full YOLO inference; it is not post-NMS reranking.
2. P3 regression and all P4/P5 logits are exactly unchanged in the trained-model audits.
3. M3 performs actual top-4 matching over 32 prototypes; all 32 are used in every fold.
4. Prototypes use outer-train images only; cache/holdout overlap is zero.
5. M3 is not convincingly better than fixed-reference M2: only mAP50-95 has a positive point estimate and its CI crosses zero.
6. M3 exceeds equal-capacity M1 by +0.0108 mAP50-95 (bootstrap CI [0.0004860332161577706, 0.02171000747815557]), supporting reference information but not matched retrieval specifically.
7. Versus E0, M3 changes mAP50/mAP50-95 by +0.0103/+0.0084; both bootstrap intervals cross zero.
8. Flash score-misranking is 16 for M3 versus 14 for M2 and 17 for E0; no robust reduction is established.
9. Low-contrast Black recall is 0.8261, above E0 0.6957 but identical to M2 0.8261.
10. Background FP clearly worsens versus E0 (10→34; bootstrap per-image CI [0.05755395683453239, 0.31884057971014496]).
11. AP75 is 0.2387, below E0 0.2442 and M2 0.2419.
12. Tiny recall is 0.8462, above E0 0.8077 but identical to M2, so it is not matching-specific.
13. All output changes originate in P3 classification scores and downstream retention/NMS; box geometry is frozen.
14. Leakage checks pass, but candidates are conservatively screened rather than expert-confirmed background; fold 0 also has a dominant prototype cluster and remains a contamination/coverage risk.
15. A matching-specific mechanism contribution is not established relative to M2.
16. Hard-negative or contrastive supervision is not added in this frozen stage because it would mask the failed matching test.
17. MBRD is not combined with CIR; the current module is not retained as the formal innovation.
