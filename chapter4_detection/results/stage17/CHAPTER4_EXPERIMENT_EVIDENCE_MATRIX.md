# Chapter 4 experiment evidence matrix

Metrics below are kept within their stated protocols; they are not pooled into one ablation.

| Method | Location | Mechanism | Protocol | mAP50-95 change | AP75 change | Judgment |
|---|---|---|---|---|---|---|
| E0 | baseline | YOLO11s | frozen 138/46, 3 seeds, best | reference, .4215±.0039 | reference | development baseline |
| E1 P2 | neck/head | P2/P3/P4/P5 detection | same 138/46, 3 seeds | +.0039 mean; variance larger | not a stable reported advantage | not retained |
| E2 Slice | input protocol | 1024 native tiles | seed42, last, different updates/checkpoint | .3716 vs .4170 | not comparable fairly | failure is confounded; abandon as formal gain |
| NBR-DE E3 | neck | learned normal-background reference/difference | 138/46, 3 seeds, best | +.0078; one seed negative | +.0532/+.0051/+.0368 | best localization pattern, but matched-init audit and weak final residual prevent core claim |
| matched NBR-DE Full | neck | matched initialization control | 138/46, 3 seeds | +.0026, mixed seeds | no robust independent claim | exploratory only |
| SFDFM | P3 neck | selective spatial-frequency fusion | 138/46, 3 seeds, best | -.0007 | -.0749 | Recall +.0453 and mAP50 +.0371, but geometry harmed; abandon |
| CBCD | P3 classification | gated local residual, regression unchanged | matched 138/46, 3 seeds | +.0156; bootstrap CI crosses 0 | -.0138 | best mean mAP50-95 candidate, unstable and no localization gain |
| CBCD OOF | P3 classification | same | source-group 3-fold, seed42, last | +.0065; CI [-.0178,.0321] | -.0038 | exploratory, not significant |
| PCRF | P3 regression | phase-consistent P2 residual | source-group 3-fold, seed42, last | +.0056; CI crosses 0 | -.0299 | stop; P2 regression fusion rejected |
| Stage12 NMS .9 | postprocess | retain overlapping candidates | frozen OOF inference | -.0181 E0 | +.0122 E0 | more GT@.75 but 2.4× duplicates; reject |
| Stage13–15 | diagnosis | TAL/DFL/native-scale audits | no new model result | n/a | n/a | candidate generation dominates; native detail alone fails |

## Supported synthesis

- Recall/coarse AP: SFDFM clearly raises mean Recall/AP50; NBR-DE and some background-aware variants show smaller/mixed improvements.
- AP75: only historical NBR-DE E3 improves AP75 in all three seeds, but its causal mechanism is not isolated. SFDFM, CBCD and PCRF do not improve it robustly.
- False positives: native full-resolution inference raises background-like predictions from 1.75 to 8.13/image; relaxed NMS greatly increases duplicates.
- Instability: E1, NBR-DE mAP, CBCD and PCRF vary by seed/fold; all quoted bootstrap intervals for their headline gains cross zero where computed.
- Stop directions: indiscriminate frequency enhancement, more P2 injection, naive slicing/native-resolution feeding, NMS relaxation, DFL-entropy ranking and endpoint masking.
- Remaining value: a classification-side, geometry-preserving mechanism that explicitly measures defect-vs-local-background evidence and knows when that reference is unreliable.

