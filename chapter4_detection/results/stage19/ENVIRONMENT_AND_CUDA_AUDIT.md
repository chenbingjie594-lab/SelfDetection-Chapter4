# Environment and CUDA audit

- `nvidia-smi`: failed to communicate with the NVIDIA driver.
- PyTorch `2.5.1+cu124`, compiled CUDA `12.4`.
- `torch.cuda.is_available()`: false; device count: 0.
- CUDA tensor allocation: `RuntimeError: No CUDA GPUs are available`; NVML initialization also failed.
- No driver, CUDA, PyTorch or system package was changed.

A real YOLO11s CPU forward with tensor shape `1×3×1536×1536` completed successfully in 0.757 s, so the required 1536 feature audit proceeded on CPU. CUDA AMP, GPU memory and synchronized GPU performance remain unavailable. Formal training would be environment-blocked even if the mechanism passed.

