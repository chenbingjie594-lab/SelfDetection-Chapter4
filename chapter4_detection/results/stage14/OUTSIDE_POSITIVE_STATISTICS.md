# Outside-positive statistics

The frozen Fold0/seed42 trace contains 6,625 positives at epochs 1, 3, 10, 25, 50, 100, and 150. An augmented GT is uniquely identified by `(epoch, assignment_call, batch_index, gt_index)`.

| Quantity | Result |
|---|---:|
| Positives outside original GT | 4,125 / 6,625 (62.26%) |
| Supervision-weight share outside | 37.75% |
| Outside positives with a negative signed edge | 99.98% |
| GT groups with at least one outside positive | 89.37% |
| GT groups with all positives outside | 20.04% |
| Tiny GT groups with all positives outside | 35.92% |
| GT groups with both inside and outside positives | 1,076 |

### Stratification

| Subset | Outside count ratio | Outside weight ratio | Outside IoU-max < .75 |
|---|---:|---:|---:|
| Flash point (class 0) | 53.37% | 33.44% | 42.17% |
| Big black spots (class 1) | 70.98% | 42.40% | 58.53% |
| Non-tiny | 41.96% | 29.71% | 14.17% |
| Tiny (`min(w,h)<8` input px) | 82.16% | 50.61% | 70.32% |
| P3 | 64.18% | 37.91% | 53.16% |
| P4 | 45.09% | 11.27% | 29.57% |
| P5 | 34.43% | 0.0002% | 33.33% |

No edge exceeded the 14.99-bin upper limit in this trace. Of all positives, 2,681 had one negative edge and 1,443 had two; 2,501 had none. The one outside row without a strictly negative edge is a floating-point boundary case.

The source image ID was deliberately not reconstructed from augmented batches: mosaic can combine multiple sources, so assigning the loader's base filename would be misleading. The trace retains a collision-free augmented-GT key. Source-level attribution remains unavailable for this run and is stated as a limitation rather than inferred.
