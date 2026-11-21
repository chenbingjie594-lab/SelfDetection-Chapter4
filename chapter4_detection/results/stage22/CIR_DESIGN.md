# CIR design

A shared 3-layer Conv-BN-SiLU encoder maps aligned 128×128 ROIs to 64 dimensions. A uses the low-resolution reconstruction, B uses original pixels, C concatenates high, low, and explicit high-minus-low features, and C-noDiff uses an identically sized zero difference input. All variants append the same 8-D deployable E0 candidate summary (box geometry, score, class, center). A zero-initialized head predicts bounded center and log-width/height residuals. E0 weights, classes, scores, candidate count and first-stage NMS remain frozen.
