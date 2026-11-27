# FSPR ablation results

| Model | P | R | mAP50 | mAP50-95 | AP75 | Flash AP | Black AP | Coverage75 | Tiny Coverage75 | BG FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0.8408 | 0.8047 | 0.8061 | 0.3540 | 0.2442 | 0.2556 | 0.4523 | 0.4598 | 0.3269 | 6 |
| F0 | 0.0565 | 0.5641 | 0.0481 | 0.0240 | 0.0224 | 0.0060 | 0.0420 | 0.5115 | 0.4038 | 21252 |
| F1 | 0.0142 | 0.6396 | 0.0128 | 0.0077 | 0.0086 | 0.0062 | 0.0092 | 0.5517 | 0.4423 | 23194 |
| F2 | 0.0225 | 0.6157 | 0.0198 | 0.0159 | 0.0252 | 0.0157 | 0.0162 | 0.5517 | 0.4231 | 22344 |

F1−F0 isolates P2 features; F2−F1 isolates the explicit phase residual; F2−E0 tests the deployed merged system. Paired intervals are in `FSPR_BOOTSTRAP.json`.
