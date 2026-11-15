# Stage 6 Final Summary

1. E3−E0 mAP50-95 is +0.0082 under paired group bootstrap, 95% CI [−0.0208, 0.0355]; uncertainty is large.
2. E3 AP75 is consistently higher by seed, but its paired CI crosses zero; target-level gains involve multiple instances but remain vulnerable to the small 58-target sample.
3. Gate/residual interventions barely change final inference metrics, so direct residual injection is weak; the training ablation indicates the main effect is likely optimization/representation learning.
4. A1 is the preferable lightweight, parsimonious core, but is not proven more accurate than E3 or E0.
5. No `PASS_UNSEEN` independent real test set was found; formal test evaluation was skipped.
6. Existing results support a cautious mechanism contribution, not a definitive generalized performance claim.
7. The critical missing evidence is a new provenance-isolated real test cohort and preregistered external replication.

No model was retrained, no historical metric was modified, and no validation/test threshold was tuned.
