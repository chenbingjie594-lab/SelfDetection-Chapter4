# MBRD implementation audit

At γ=0, module output is bit-identical to input. Unit tests cover all modes, vectorized nearest-prototype retrieval and query-independent M2. Formal trained-model audits record exactly unchanged P3/P4/P5 regression logits and P4/P5 classification logits, while trained P3 classification logits change. Full YOLO inference—not post-NMS re-ranking—produces all formal predictions.
