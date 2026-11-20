# Stage18 final summary

1. RGLCD was correctly integrated as a persistent custom Detect head; P3 classification alone receives it.
2. Matched E0 initialization is exact for 499 shared tensors; raw scores, raw boxes and decoded outputs are exactly equal at gamma zero.
3. Real YOLO-loss auditing confirms gamma updates immediately and phi receives nonzero finite gradients after gamma moves.
4. Train-only GT features have much stronger standardized local contrast than sampled background (`|D|` 3.63 vs .79).
5. The reliability term does not perform its intended role: GT/background both have `r≈.989`; low-contrast Black safety is not sufficiently established.
6. Background false positives, mAP50-95, AP75, tiny high-quality candidate coverage and trained inference contribution are unavailable because formal CV was stopped.
7. No fold consistency or bootstrap claim exists.
8. RGLCD has verifiable implementation differences from NBR-DE/CBCD, but its ingredients overlap established local-contrast/variance-weighting literature and originality is not claimed.
9. Decision: **FAIL_PRETRAIN_MECHANISM_GATE — do not run multi-seed or ablations and do not tune alpha/window/eps in this round.**

The code and negative mechanism result are retained for reproducibility. No data, annotation, historical weight or historical result was modified.

