# FSPR negative sampling audit

R1 uses N64 natural sampling; R2/R3 use N256 natural sampling. All share 19,772 samples/epoch and 3,120 updates. N256 exposes every fit sample exactly 40 times. No WeightedRandomSampler is used. High-response unmatched E0 regions remain UNREVIEWED and are excluded from trusted negatives.
