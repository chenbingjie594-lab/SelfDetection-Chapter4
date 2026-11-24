# MBRD ablation results

| Model | P | R | mAP50 | mAP50-95 | AP75 | Flash AP | Black AP | Low-contrast Black R | Tiny R | BG FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0.8408 | 0.8047 | 0.8061 | 0.3540 | 0.2442 | 0.2556 | 0.4523 | 0.6957 | 0.8077 | 10 |
| M1 | 0.7927 | 0.8345 | 0.8115 | 0.3516 | 0.2232 | 0.2605 | 0.4426 | 0.7826 | 0.8654 | 36 |
| M2 | 0.8005 | 0.8413 | 0.8179 | 0.3593 | 0.2419 | 0.2653 | 0.4532 | 0.8261 | 0.8462 | 28 |
| M3 | 0.8148 | 0.8184 | 0.8164 | 0.3624 | 0.2387 | 0.2636 | 0.4612 | 0.8261 | 0.8462 | 34 |

M3−M2 isolates query-dependent texture matching; M3−M1 tests background reference beyond equal-capacity ordinary enhancement. Source-group intervals are in `MBRD_BOOTSTRAP.json`.
