# RGLCD implementation audit

- `RGLCDDetect` replaces the final Detect explicitly; `cv2`, `cv3`, `dfl`, strides and metadata are copied.
- Only P3 `cv3` consumes RGLCD output; P3 `cv2` and all P4/P5 paths consume originals.
- Save/reload retains `RGLCDDetect`.
- 499 shared state tensors equal the matched E0 initialization exactly.
- New state: scalar gamma, nonzero 128×128 phi, fixed 7×7 ring mask.
- At gamma zero, raw boxes, raw scores and decoded output all have maximum absolute difference 0 on the same real input/mode.
- Effective border count ranges from 12 to 40; padding zeros are excluded. Variance is clamped nonnegative and statistics use float32.
- Unit tests cover finite constant/FP16 inputs. CUDA AMP, CUDA validator and trainer-reload runtime checks remain blocked because the NVIDIA driver is unavailable.

