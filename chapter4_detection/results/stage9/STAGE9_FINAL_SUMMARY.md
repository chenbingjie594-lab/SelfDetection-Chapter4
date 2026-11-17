# Stage 9 Final Summary

## Answers

1. SFDFM's AP75 decline is primarily geometric. E0 boxes with SFDFM scores retained AP75=0.4027, while SFDFM boxes with E0 scores reached only 0.3649 versus E0's 0.4140.
2. Local classification/regression gradient disagreement is observable, but mean cosine is positive and only 25.6% of E0 observations are negative. This does not establish persistent harmful conflict.
3. Negative cosine was not concentrated in small or low-contrast training samples.
4. A classification-only enhancement was reasonable as one frozen diagnostic candidate because SFDFM scores provided small benefit on E0 boxes. It is not shown to be generally necessary.
5. CBCD avoided SFDFM's severe AP75 collapse and increased mean mAP50-95, but mean AP75 remained below E0 and low-contrast Black did not improve repeatably.
6. CBCD did not show a positive mAP50-95 difference in every seed; bootstrap uncertainty crossed zero. Cross-seed benefit is insufficiently established.
7. The technical placement differs from whole-feature enhancement by restricting local-contrast gating to P3 classification while copying the regression tower unchanged. The internal residual/gating operations overlap substantially with existing contrast and attention designs, limiting originality claims.
8. Evidence is insufficient to present CBCD as the formal fourth-chapter innovation. It is a useful negative/mixed mechanism study.

The current evidence points to a coupled data-limited localization and background-discrimination problem. Neither whole-feature frequency fusion nor classification-only contrast enhancement provides stable, statistically supported improvement. Further variants should not be searched on this validation split.
