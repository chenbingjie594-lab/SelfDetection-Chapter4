# Pre-holdout mechanism diagnosis

Scope was frozen Fold0 **training only**: 92 images, 115 GT, feature diagnosis at 640 because CUDA/1536 was unavailable. Four deterministic non-GT grid backgrounds per image produced 368 controls.

| group | n | mean abs(D) | mean signed D | mean V | mean r | projected residual abs |
|---|---:|---:|---:|---:|---:|---:|
| GT | 115 | 3.630 | 2.634 | .01126 | .98899 | 5.930 |
| background | 368 | .792 | -.049 | .01074 | .98944 | .718 |
| Flash | 54 | 4.015 | 2.977 | .01181 | .98842 | 6.478 |
| Black | 61 | 3.289 | 2.329 | .01076 | .98950 | 5.445 |

The standardized contrast contains strong train-only discriminative signal. However, reliability is almost identical for GT/background and 46.1% of GT values exceed .99. With alpha=1 the reliability path is near-identity at this feature scale; it does not demonstrably suppress unreliable periodic texture. The grid control is not a comprehensive hard-texture sampler, and low-contrast Black was not separately identifiable in this feature export, so over-suppression cannot be claimed absent. Ring/GT overlap proxy was zero at this diagnostic scale.

**Predeclared decision:** numerical learning passes, but the defining reliability mechanism fails its pre-holdout usefulness gate. Do not read Fold holdout labels or launch three-fold training. Changing alpha after this observation would be a new, separately frozen study, prohibited in this round.

