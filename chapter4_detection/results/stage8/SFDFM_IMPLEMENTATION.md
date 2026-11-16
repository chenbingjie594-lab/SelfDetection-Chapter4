# SFDFM Implementation

- `models/sfdfm.py`: exact fixed Haar DWT/IDWT, selective detail/difference fusion and runtime registration.
- `models/yolo11s_sfdfm.yaml`: original YOLO11s topology plus one P4-to-P3 SFDFM route; original P4/P5 path preserved.
- `scripts/stage8_model_audit.py`: explicit layer mapping from each Stage7 matched initialization, tensor equality, initial output/loss and branch-gradient audit.
- `tests/test_sfdfm.py`: even/odd DWT reconstruction, output shape, FP32/AMP stability, first-backward gradients, finite values, YAML registration and checkpoint reload.

Audit per seed: 499/499 source tensors and all 9,458,631 baseline elements transferred; zero shared-value mismatches. Initial raw-output relative L2 is 0.00743/0.00742/0.00722 for seeds 42/3407/2026. Every new trainable parameter has a finite non-zero gradient on the first real YOLO-loss backward pass.

The 1-epoch smoke run completed training, validation, AMP checking and checkpoint reload at imgsz1536. A one-epoch metric of zero is expected during warmup and is not reported as an experiment result.
