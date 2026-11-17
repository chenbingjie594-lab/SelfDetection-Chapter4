# Classification–Regression Gradient Analysis

The installed differentiable `v8DetectionLoss` vector was used directly: `L_cls` and `L_box+L_dfl` were differentiated separately with respect to the shared Neck P3 tensor. Detached log items were not used. All 138 frozen training images and three seeds were analyzed; validation images were excluded.

At trained best checkpoints (414 image/checkpoint observations per model):

| Model | Mean cosine | Negative fraction | Near-zero |
|---|---:|---:|---:|
| Matched-E0 | 0.0426 | 25.6% | 0 |
| SFDFM | 0.0506 | 26.8% | 0 |

Matched-E0 negative fractions by seed were 21.7%, 28.3%, 26.8%; SFDFM values were 21.7%, 37.0%, 21.7%. This is not a stable SFDFM-induced shift.

Using the train-only median normalized area, E0 negative fractions were 24.6% for small and 26.6% for non-small images. Using the train-only median local contrast from Stage 8, they were 22.6% for low-contrast and 29.4% otherwise. Thus negative gradients were not concentrated in small or low-contrast samples.

Initial checkpoints had cosines extremely close to zero and seed-sensitive negative fractions, plus one near-zero sample per seed. Because magnitude and direction evolve strongly during training, a negative sign alone is not evidence of harmful conflict.

Conclusion: local classification/regression direction disagreement is observable, but there is no evidence of persistent globally negative conflict or concentration in the proposed difficult groups. CBCD is therefore motivated by the crossover result (classification information can help while box geometry should be preserved), not by a claim that gradient conflict has been proven harmful.
