# YOLO11s Detection Head Audit

## Frozen environment

- Ultralytics 8.4.145, YOLO11s detection checkpoints from Stage 7/8.
- Installed implementation: `ultralytics/nn/modules/head.py::Detect`.
- Loss: `ultralytics/utils/loss.py::v8DetectionLoss`.

## Structure

Each P3/P4/P5 Neck feature is supplied to two already separate towers:

```text
Neck Fi ─┬─ cv2[i]: Conv3x3 → Conv3x3 → Conv1x1(4×reg_max) ─ DFL/decode ─ box
         └─ cv3[i]: DWConv3x3+Conv1x1 → DWConv3x3+Conv1x1 → Conv1x1(nc) ─ sigmoid ─ class score
```

For this model `nc=2`, `reg_max=16`, so every candidate has 64 raw distribution-regression channels and 2 classification logits. The three scales use strides 8/16/32. Raw scale tensors are flattened in P3→P4→P5 order; DFL converts each four 16-bin distribution to distances from a fixed anchor point, followed by `dist2bbox` and stride scaling. The standard checkpoint is not end-to-end and has no active one-to-one head.

The towers are distinct, but both receive the same Neck feature `Fi`. Consequently `L_cls` backpropagates through `cv3` into Fi and the shared Neck/Backbone, while `L_box+L_dfl` backpropagates through `cv2` into the same Fi and shared upstream parameters. Separate towers do not imply gradient-isolated feature learning.

For a representative validator-shaped 384×1536 tensor, both Matched-E0 and SFDFM produced exactly 12,096 aligned candidates:

| Scale | Stride | Feature shape | Candidates |
|---|---:|---|---:|
| P3 | 8 | 1×128×48×192 | 9,216 |
| P4 | 16 | 1×256×24×96 | 2,304 |
| P5 | 32 | 1×512×12×48 | 576 |

Both raw outputs were boxes `1×64×12096` and logits `1×2×12096`, with identical scale ordering, strides, class count and `reg_max`. The Detect head contains 820,182 parameters: 639,936 in `cv2`, 180,230 in `cv3`, and 16 fixed DFL projection elements. This establishes structural index compatibility; the image-by-image preprocessing/anchor equality gate is still checked before crossover evaluation.

## Loss and inference

`v8DetectionLoss` uses TaskAlignedAssigner with detached class probabilities and detached decoded boxes to construct targets. Its differentiable terms are BCE classification, IoU box loss and DFL. Gains are applied before returning the three-element differentiable loss vector. At inference, decoded boxes and sigmoid class scores are concatenated before fixed confidence filtering and class-aware NMS.

## Innovation boundary

YOLO11s already separates classification and regression towers. Merely describing or duplicating this separation is not an innovation. A Stage-9 candidate would have to alter only the classification feature input while retaining the original regression input path, and must be distinguished from established residual/local-contrast gating methods. Even then, originality and performance remain empirical questions.
