# HBD design

HBD adds a training-only 32-D normalized projection and class-conditional positive-versus-hard-background prototype margin loss (`m=0.1`, `tau=0.1`, `lambda=0.1`) to the frozen P3 MBRD adapter. H1 is matched-reference BCE only; H2 is fixed-reference HBD; H3 is matched-reference HBD; H4 replaces class-conditioned prototypes by an equal-capacity generic contrastive control. Projection heads are absent from inference.
