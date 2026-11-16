# Initialization and Training Audit

Historical E0 and E3 did not use a strictly identical 2-class initialization path. E0 let the Ultralytics trainer adapt the 80-class checkpoint; E3 built a 2-class YAML model before training and explicitly transferred semantic tensors. Historical E3 left the six class-prediction tensors plus NBR-DE tensors randomly initialized. Both runs selected AdamW automatically at lr=0.001667, but E0 used parameter groups 81/88/87 and E3 used 82/95/89 (no-decay/decay/bias). Consequently historical E3−E0 differences cannot be treated as initialization-controlled causal effects.

For Stage 7, each seed received one matched 2-class E0 initialization; Identity and Full copied every shared tensor, including classifier/regressor heads, BatchNorm parameters and running statistics.

| Seed | Shared parameter mismatches (Identity/Full) | Raw output max abs (Identity/Full) | Loss delta (Identity/Full) | Shared gradient mismatches (Identity/Full) |
|---:|---:|---:|---:|---:|
| 42 | 0/0 | 0.0e+00/0.0e+00 | 0.0e+00/0.0e+00 | 0/0 |
| 3407 | 0/0 | 0.0e+00/0.0e+00 | 0.0e+00/0.0e+00 | 0/0 |
| 2026 | 0/0 | 0.0e+00/0.0e+00 | 0.0e+00/0.0e+00 | 0/0 |

All matched checks must be zero before formal training. Checkpoint SHA256 values are stored in each audit JSON; checkpoint binaries are excluded from Git.
