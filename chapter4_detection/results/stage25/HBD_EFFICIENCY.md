# HBD efficiency

| Variant | Inference adapter parameters | Training-only projection | Mean observed CPU seconds/original |
|---|---:|---:|---:|
| H1 | 16,449 | 0 | 0.576 |
| H2 | 16,449 | 4,096 | 0.572 |
| H3 | 16,449 | 4,096 | 0.564 |
| H4 | 16,449 | 4,096 | 0.547 |

The projection head is omitted at inference. Timings were measured while 12 CPU inference jobs ran concurrently under uncontrolled system load, so they demonstrate executability rather than reliable latency ranking. CUDA peak memory is unavailable because the NVIDIA driver could not be queried. H2 uses fixed reference; H1/H3/H4 retain matched-reference retrieval and therefore have the same inference graph as Stage24 M3.
