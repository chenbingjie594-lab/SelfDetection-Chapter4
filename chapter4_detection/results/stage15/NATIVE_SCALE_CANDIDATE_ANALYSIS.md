# Native-scale candidate analysis

All conditions use the same three Stage10 Matched-E0 seed42 `last.pt` OOF checkpoints, class score ≥0.001, class-aware NMS IoU 0.7, and max_det 300. B1 and B2 both use exactly `1×3×1024×4096`; B2 differs only by frozen 1536×384 downsampling followed by 4096×1024 restoration.

| Condition | Mean Qmax | GT ≥.50 | GT ≥.75 | GT ≥.90 | Tiny GT ≥.75 |
|---|---:|---:|---:|---:|---:|
| B0 1536 square | 0.739 | 95.40% | 52.87% | 5.17% | 34.62% |
| B1 native detail | 0.387 | 32.18% | 4.02% | 0.00% | 9.62% |
| B2 restored detail | 0.386 | 31.03% | 4.60% | 1.15% | 11.54% |

At IoU≥.75, B1 covers 3.70% of Flash and 4.30% of Black GT; B2 covers 2.47% and 6.45%. B1's mean Qmax advantage over B2 is only +0.00082; for the 52 tiny GT it is +0.0294, but B1 still covers one fewer tiny GT at .75. B1 does not improve center/width/height errors overall.

Final predictions per image rise from 5.39 (B0) to 10.12 (B1) and 13.47 (B2). Approximate background predictions with maximum GT IoU below .1 rise from 1.75 to 8.13 and 11.37 per image. Native input therefore causes severe training–inference scale shift and background activation.

The result separates “more pixels” from “useful candidates”: the unchanged model cannot exploit native input directly. B1 and B2 are nearly identical at high-quality candidate generation, so this experiment provides no stable original-detail benefit.
