# CIR implementation audit

All four variants initialize with exactly zero residuals and E0-identical boxes. Unit tests verify head gradients, crop geometry, residual decoding, save/reload, and CPU forward/backward. Every 40-epoch run has nonzero gradients and falling loss. No negative-size box is possible because width/height use exponentiated bounded log residuals; inference clips image boundaries. CUDA was unavailable, so all local training and inference used CPU.
