# Stage21 mechanism decision

**Decision: `NO_STABLE_SIGNED_RESIDUAL_SIGNAL`. Do not implement or formally train SCR.**

The best fixed deployable probe misses the required 10% MAE improvement (4.52%), its source-group bootstrap interval crosses zero, mean candidate IoU does not improve, and IoU≥0.75 coverage decreases. Although four of five folds improve residual MAE and both axes move in the desired direction, the effect is not large or operationally consistent enough to justify a network module.

Consequently no `SCR_FEASIBILITY_AND_DESIGN.md`, `SCR_IMPLEMENTATION_AUDIT.md`, model file, checkpoint or formal metric is generated. A plain MLP width/height correction would also have weak novelty relative to ordinary box refinement; renaming it SCR would not solve that problem. Existing YOLO loss semantics would require explicit verification that a decoded-box correction participates in training, but this question is not pursued because the evidence gate failed.

CUDA is additionally unavailable (`torch.cuda.is_available()==False`, device count 0, CUDA tensor allocation fails, and `nvidia-smi` cannot communicate with the driver), but CUDA is not the primary stop reason.
