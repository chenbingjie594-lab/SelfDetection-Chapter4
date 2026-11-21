# QPR-CIR ablation

| Model | P | R | mAP50 | mAP50-95 | AP75 | Flash AP | Black AP | Coverage75 | Tiny cov. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0.8408 | 0.8047 | 0.8061 | 0.3540 | 0.2442 | 0.2556 | 0.4523 | 80/174 | 17/52 |
| C | 0.8755 | 0.8477 | 0.8610 | 0.3743 | 0.2506 | 0.3047 | 0.4438 | 82/174 | 17/52 |
| C-Half | 0.8568 | 0.8294 | 0.8390 | 0.3741 | 0.2328 | 0.2870 | 0.4613 | 82/174 | 16/52 |
| C-GateGeo | 0.8568 | 0.8294 | 0.8390 | 0.3719 | 0.2405 | 0.2821 | 0.4616 | 84/174 | 16/52 |
| C-GateFull | 0.8568 | 0.8294 | 0.8390 | 0.3725 | 0.2270 | 0.2825 | 0.4625 | 82/174 | 15/52 |
| Oracle-Gate | 0.8813 | 0.8531 | 0.8715 | 0.4202 | 0.3398 | 0.3333 | 0.5072 | 95/174 | 20/52 |

C-Half isolates uniform attenuation; GateGeo tests deployable geometry-only selection; GateFull tests whether cross-resolution summaries add quality-prediction value. Paired intervals are in `QPR_CIR_BOOTSTRAP.json`.
