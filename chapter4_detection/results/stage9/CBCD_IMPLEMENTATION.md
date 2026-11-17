# CBCD Implementation

`models/cbcd.py` provides `CBCD`, `CBCDDetect`, explicit head replacement and `CBCDTrainer`. No source-string patching or site-packages modification is used. The custom trainer is necessary because stock Ultralytics reconstructs a plain Detect from the checkpoint YAML; the first smoke test intentionally failed the reload audit and is retained locally as `e6_cbcd_seed42_smoke_failed_plainhead`.

The corrected smoke test passed AMP checks, one training epoch, validation, checkpoint saving and reloading. Reloaded model summary retained the custom head (110 fused layers, 9,432,262 fused parameters, 21.7 GFLOPs). The zero one-epoch AP is expected from the matched pre-training initialization and is not a formal result.

Tests at freeze: 43 passed, 3 skipped. Formal training must use `CBCDTrainer`; `train.py` rejects E6 checkpoints lacking `CBCDDetect`.
