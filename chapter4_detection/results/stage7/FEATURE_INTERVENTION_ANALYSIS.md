# Feature Intervention Analysis

| Bypass | P3 δ | P4 δ | P5 δ | Class-logit δ | Box-raw δ | Decoded-candidate δ | Defect/background change |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate_bypass | 6.546e-04 | 3.298e-04 | 2.287e-04 | 8.825e-05 | 1.997e-04 | 1.141e-05 | 3.34× |
| residual_bypass | 6.040e-04 | 4.144e-04 | 4.255e-04 | 7.322e-05 | 1.921e-04 | 1.068e-05 | 2.71× |

The residual has a measurable, defect-concentrated effect, but its magnitude is about 10⁻⁴–10⁻³ in intermediate features and around 10⁻⁵ after decoding. It is therefore attenuated rather than exactly absent. Gamma was not artificially changed.

At confidence 0.25, all 171 detections across 3 seeds × 46 images matched one-to-one under both interventions; no final detection appeared or disappeared. Gate bypass changed confidence by mean `1.667e-04` and produced matched-box IoU `0.999791`; residual bypass changed confidence by `1.488e-04` with IoU `0.999905`.
