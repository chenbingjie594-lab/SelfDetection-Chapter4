# Multi-Scale Feature Diagnosis

## Protocol

All 138 frozen training images were evaluated with the three Stage7 Matched-E0 best checkpoints. GT boxes were mapped through the fixed 1536 letterbox transform. Feature response inside each HBB was compared with an expanded non-box ring at Backbone P2/P3 and Neck P3/P4/P5. Confidence intervals are source-group bootstrap intervals.

## Defect/background feature contrast

| Layer | All | Flash | Black |
|---|---:|---:|---:|
| Backbone P2 | 0.617 [0.590, 0.645] | 0.559 [0.527, 0.589] | 0.668 [0.630, 0.708] |
| Backbone P3 | 0.089 [0.069, 0.110] | 0.057 [0.038, 0.075] | 0.117 [0.084, 0.149] |
| Neck P3 | 1.131 [1.098, 1.164] | 1.096 [1.039, 1.156] | 1.161 [1.098, 1.220] |
| Neck P4 | 0.265 [0.220, 0.309] | 0.229 [0.198, 0.267] | 0.265 [0.220, 0.309] |
| Neck P5 | -0.048 [-0.060, -0.037] | -0.046 [-0.056, -0.036] | -0.048 [-0.060, -0.037] |

The raw data JSON/CSV should be used for exact class values; rounded table values summarize the stable pattern. Downsampling from P2 to Backbone P3 sharply reduces local contrast. Contrary to the “Neck dilution at P3” hypothesis, Neck P3 strongly restores separability in all seeds/classes. Separability then drops at P4 and becomes slightly negative at P5.

This supports a small intervention at Neck P3 that uses P4 context without changing the existing detection scales. It does not support adding another P2 head or replacing the backbone. Because Neck P3 is already strong, SFDFM must be treated as a high-risk, low-margin refinement rather than an assumed improvement.
