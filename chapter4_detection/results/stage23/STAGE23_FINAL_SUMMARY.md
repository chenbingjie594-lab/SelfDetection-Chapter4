# Stage23 final summary

| Model | P | R | mAP50 | mAP50-95 | AP75 | Flash AP | Black AP | Coverage75 | Tiny cov. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0.8408 | 0.8047 | 0.8061 | 0.3540 | 0.2442 | 0.2556 | 0.4523 | 80/174 | 17/52 |
| C | 0.8755 | 0.8477 | 0.8610 | 0.3743 | 0.2506 | 0.3047 | 0.4438 | 82/174 | 17/52 |
| C-Half | 0.8568 | 0.8294 | 0.8390 | 0.3741 | 0.2328 | 0.2870 | 0.4613 | 82/174 | 16/52 |
| C-GateGeo | 0.8568 | 0.8294 | 0.8390 | 0.3719 | 0.2405 | 0.2821 | 0.4616 | 84/174 | 16/52 |
| C-GateFull | 0.8568 | 0.8294 | 0.8390 | 0.3725 | 0.2270 | 0.2825 | 0.4625 | 82/174 | 15/52 |
| Oracle-Gate | 0.8813 | 0.8531 | 0.8715 | 0.4202 | 0.3398 | 0.3333 | 0.5072 | 95/174 | 20/52 |

Frozen QPR-CIR decision: **STOP**. This joint point-estimate rule does not imply statistical significance; source-group intervals and fold direction must be considered. Stage22 remains STOP regardless of this correction.

## Required conclusions

1. The Stage22 coverage evaluator is corrected: it now independently re-selects the best same-class candidate after refinement.
2. Corrected CIR-C coverage is 82/174 versus E0 80/174: 15 GT gain and 13 GT lose coverage, a net +2.
3. Harmful corrections concentrate among already adequate/high-quality boxes; C loses 13 high-quality GT despite its 15 gains.
4. Inference-available features show only modest quality separation. GateFull mean g is 0.4105 for useful versus 0.3717 for harmful candidates.
5. GateFull is not better than GateGeo: mAP50-95 0.3725 versus 0.3719, AP75 0.2270 versus 0.2405.
6. GateFull is not better than C-Half: mAP50-95 0.3725 versus 0.3741, AP75 0.2270 versus 0.2328.
7. GateFull reduces high-quality losses from 13 to 5, but also reduces gains from 15 to 7; coverage remains 82/174.
8. GateFull AP75 falls below C (0.2270 versus 0.2506); mAP50-95 is also slightly lower (0.3725 versus 0.3743).
9. Flash AP falls relative to C (0.2825 versus 0.3047), while Black AP improves (0.4625 versus 0.4438); both classes do not jointly benefit.
10. Tiny coverage degrades from C 17/52 to GateFull 15/52.
11. Background-style count stays 322; duplicate-GT count is 106 versus C 104.
12. The diagnostic Oracle-Gate shows headroom (mAP50-95 0.4202, AP75 0.3398), but it uses GT and is not a deployable model.
13. Cross-resolution quality features do not establish an advantage over geometry-only gating. A generic sigmoid quality gate is not itself novel.
14. QPR-CIR should stop as a formal Chapter-4 innovation candidate under the frozen Stage23 criteria.
