# Annotation uncertainty versus regression error

For a 1 px coordinate change, the median relative width/height change is 0.0385/0.0400 across all train GT and 0.0500/0.0526 for tiny GT. The corresponding 2 px tiny ratios exceed 0.100/0.105. Stage20 median absolute width/height prediction errors were about 0.091/0.093, so one-to-two original pixels are of a comparable relative order for many tiny boxes. This comparison does not estimate real label noise.

The correlation between per-instance one-pixel relative sensitivity and frozen oracle width/height absolute error is 0.240; it is descriptive and does not establish causality. Near-border instances are protected by clipping in the simulation and no official labels were edited.

Historical provenance review traced 149 XML/evidence mappings; 2 were below IoU 0.99. All 27 Stage17 expert-review entries remain `PENDING_EXPERT_REVIEW`. Existing confirmed provenance and unresolved human boundary review are therefore reported separately. There is no new evidence that the labels are wrong—only strong evidence that high-IoU evaluation is pixel-sensitive.
