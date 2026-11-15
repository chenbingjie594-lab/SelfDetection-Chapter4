# NBR-DE Gate Intervention

No retraining was performed. Each E3 best checkpoint was evaluated under Full, Gate Bypass (`F+γD`), and Residual Bypass (`F`) using identical preprocessing/NMS.

| Seed | Mode | mAP50-95 | AP75 | Black AP | Black Recall | Black TP/FP/FN @.25 | Low-contrast Black | Background FP @.25 |
|---:|---|---:|---:|---:|---:|---|---:|---:|
| 42 | full | 0.4377 | 0.4031 | 0.4344 | 0.6646 | 20/3/11 | 3/13 | 1 |
| 42 | gate_bypass | 0.4372 | 0.4031 | 0.4346 | 0.6645 | 20/3/11 | 3/13 | 1 |
| 42 | residual_bypass | 0.4378 | 0.4032 | 0.4346 | 0.6645 | 20/3/11 | 3/13 | 1 |
| 3407 | full | 0.4168 | 0.3665 | 0.4218 | 0.7213 | 22/5/9 | 5/13 | 3 |
| 3407 | gate_bypass | 0.4164 | 0.3665 | 0.4218 | 0.7210 | 22/5/9 | 5/13 | 3 |
| 3407 | residual_bypass | 0.4168 | 0.3665 | 0.4218 | 0.7213 | 22/5/9 | 5/13 | 3 |
| 2026 | full | 0.4335 | 0.4228 | 0.4174 | 0.6272 | 20/5/11 | 3/13 | 3 |
| 2026 | gate_bypass | 0.4326 | 0.4181 | 0.4174 | 0.6274 | 20/5/11 | 3/13 | 3 |
| 2026 | residual_bypass | 0.4320 | 0.4145 | 0.4174 | 0.6272 | 20/5/11 | 3/13 | 3 |

Gate Bypass changes mAP50-95 by less than 0.001 for every seed. Residual Bypass is likewise nearly unchanged. Therefore the learned residual injection has little direct inference-time effect at the final checkpoints. Combined with the much poorer separately trained No-Gate ablation, the evidence is most consistent with the gate/module affecting training optimization or representation learning while the final explicit residual magnitude is small. The intervention creates distribution shift and is mechanism evidence, not standalone causal proof.
