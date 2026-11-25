# Stage24 diagnostic correction

Stage24's historical `BG FP` counted confidence ≥0.25 predictions whose maximum IoU to a **same-class** GT was below 0.10. It therefore mixed true background-style predictions with class confusion. Re-evaluation with one fixed Stage25 post-NMS taxonomy gives:

| Model | Stage24 legacy count | Background style (any-class IoU<0.10) | Class confusion | Same-class localization | Duplicate |
|---|---:|---:|---:|---:|---:|
| E0 | 10 | 6 | 4 | 17 | 4 |
| M2 | 28 | 24 | 4 | 16 | 25 |
| M3 | 34 | 30 | 4 | 16 | 24 |

Localization and duplicate counts are separate unmatched-prediction categories and are not arithmetically part of the legacy count. The new fixed priority is correct one-to-one match → duplicate → class confusion → same-class localization → background style → unconfirmed overlap. This is a uniform post-NMS diagnostic for all models; unlike Stage12 TraceableNMS, it does not claim the exact suppressing candidate.

Stage24 cached patches also used replicate padding. At real P3 horizontal boundaries this was not equivalent to the frozen full-map classification tower. Stage24 results remain historically reproducible and are not overwritten, but the limitation is now explicit. Stage25 uses zero padding plus validity masks after every spatial classifier block.
