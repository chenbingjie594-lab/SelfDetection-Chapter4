# High-IoU Localization Diagnosis

Per-GT analysis used original-image coordinates and the highest-IoU same-class prediction for descriptive geometry. Counts at IoU 0.50/0.75/0.90 were:

| Seed | E0 | SFDFM |
|---:|---|---|
| 42 | 53 / 35 / 5 | 52 / 28 / 2 |
| 3407 | 51 / 32 / 4 | 54 / 32 / 3 |
| 2026 | 49 / 29 / 6 | 52 / 30 / 4 |

SFDFM often added coarse IoU≥0.50 matches, but reduced IoU≥0.90 matches in every seed. On common IoU≥0.50 detections, width/height errors were not consistently improved and high-IoU set membership changed substantially. These results agree with the crossover intervention: the AP75 degradation is predominantly geometric rather than solely a confidence-ranking artifact.

The instance CSV files retain center, width, height, signed area, confidence, border distance and match-status fields. Group diagnostics from Stage 8 showed no repeatable improvement for low-contrast or small Black targets.
