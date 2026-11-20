# Efficiency status

| model | parameters | delta |
|---|---:|---:|
| matched E0 | 9,428,566 | — |
| RGLCD | 9,444,951 | +16,385 (+0.174%) |

For square 1536 input, P3 is 192×192 with 128 channels. Two fixed ring moment operations plus the 1×1 projection add approximately 0.98 G MACs, or 1.96 GFLOPs under the two-FLOP/MAC convention, excluding inexpensive elementwise operations. This is an analytical estimate and is labelled as such.

Fair CUDA latency, peak memory, synchronized throughput and profiler FLOPs were not measured because the NVIDIA driver was unavailable. Training-log timing is not substituted.

