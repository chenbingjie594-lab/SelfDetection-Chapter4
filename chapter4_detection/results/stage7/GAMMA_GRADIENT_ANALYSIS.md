# Gamma Gradient Analysis

The real YOLO detection loss confirms the analytic routing. At gamma=0, gamma receives a non-zero gradient while Q/K/V, difference and gate gradients are exactly zero; adjacent main-network layers retain non-zero gradients. After the first successful optimizer update makes gamma non-zero, all active NBR-DE branches receive gradients.

AMP initially skipped overflowing attempts and kept gamma at zero. The first successful AMP update occurred only after GradScaler reduced its scale; skipped attempts are preserved rather than counted as optimizer updates.

| Precision | Successful update | gamma before | gamma after | gamma grad | Q grad | Difference grad | Gate grad |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 0 | 0 | 0.001 | 1.69 | 0 | 0 | 0 |
| FP32 | 1 | 0.001 | 0.00114564 | 1.124 | 0.0001174 | 0.03497 | 0.00597 |
| AMP | 0 | 0 | 0.001 | 1.811 | 0 | 0 | 0 |
| AMP | 1 | 0.001 | 0.00177072 | 0.2649 | 0.000101 | 0.03148 | 0.005322 |
