# MBRD design

MBRD retrieves the top-4 of 32 fold-specific normalized conservative-background prototypes by vectorized cosine similarity (τ=0.1), forms `[q, q-r, q⊙r]`, and applies a 3C→32→C 1×1 residual adapter with zero-initialized scalar γ. It is inserted only before the frozen P3 classification tower. M1 uses the same adapter capacity without a reference; M2 uses one deterministic query-independent mean reference; M3 uses matched retrieval.
