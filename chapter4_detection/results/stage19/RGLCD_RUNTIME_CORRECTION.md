# RGLCD runtime correction

`diagnostics_enabled` now defaults to false. `last_stats` CPU scalar conversions execute only when explicitly enabled; formal forward computation is unchanged. `getattr(..., false)` preserves compatibility with the Stage18 pickled checkpoint created before the flag existed.

Tests confirm diagnostics-off/on equality for output and scalar loss, and exact equality of input, gamma and phi gradients. Gamma/phi are not detached. Constant FP16 input remains finite. Stage18 checkpoint loading and `RGLCDDetect` persistence pass. CUDA AMP could not be executed because no CUDA device is visible; it is not claimed as tested.

