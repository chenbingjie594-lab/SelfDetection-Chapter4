# CBCD Candidate Design

CBCD is a conditional candidate, not a claimed innovation. It leaves the P3 regression input unchanged and augments only the P3 classification input with a depthwise local residual, absolute residual projection and learned gate. P4/P5, DFL, box towers and the loss remain unchanged.

This construction overlaps conceptually with established local-contrast residual and gated attention methods. Its project-specific distinction is the strict placement inside only the existing YOLO classification tower, motivated by a preregistered box–score crossover. That placement alone is not sufficient to establish broad methodological originality.

The projection is initialized with small nonzero weights. Across all seeds, raw regression outputs were bitwise identical for the same P3/P4/P5 inputs, classification-logit relative changes were approximately 4–5×10⁻⁵, and every new trainable parameter received a finite nonzero first-backward gradient.
