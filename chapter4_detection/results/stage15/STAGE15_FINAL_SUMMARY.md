# Stage15 final summary

## Required answers

1. Down/up sampling measurably removes gradients and Laplacian energy and changes GT-boundary pixels; defect Laplacian response falls 78.94%.
2. No: matched background high-frequency attenuation is generally larger, and normalized defect/background response contrast does not systematically decline.
3. E2 failure is not a coverage failure. Confirmed confounders are 71.45% background tiles, duplicated GT exposure, reduced global context, 1,723 vs 184 update attempts, 150-last vs 76-best checkpoint protocol, and different input/Mosaic geometry. Their individual effects are hypotheses.
4. No. B1 IoU≥.75 candidate coverage is 4.02%, versus B0 52.87%.
5. No stable evidence. B1 and B2 have nearly identical candidate and AP results; source-group intervals cross zero.
6. No. Native-scale center and size errors are much worse because of severe scale mismatch.
7. Yes. B1 background-like predictions increase from 1.75 to 8.13/image, time rises 1.24×, and memory 2.30× versus B0.
8. Both classes collapse. B1 Flash/Black AP50-95 are 0.0054/0.0246; neither class establishes native-detail value.
9. Not yet. The predeclared evidence gates for a global-context/local-detail E9 are not met, so no E9 was implemented.
10. A testable future hypothesis remains: native detail may help only after scale-matched local feature extraction constrained by global context. Stage15 does not provide enough positive evidence to promote it as the next model innovation.

## Decision

**Do not develop E9 in this stage.** Merely feeding the frozen detector more pixels creates distribution shift, background false positives, and poor geometry. The B1–B2 controlled comparison does not show that retained original detail adds reliable high-IoU candidates. The next defensible work is expert annotation review and, only with new evidence, a single frozen scale-matched global/local hypothesis—not repeated OOF architecture search.

All results use real images only. No frozen data, historical result, checkpoint, TAL, DFL, backbone, neck, or head was modified.
