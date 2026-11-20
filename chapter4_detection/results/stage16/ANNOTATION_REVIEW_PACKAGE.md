# Stage16 annotation review package

## Status

- 25/25 review records and plates were rebuilt from the frozen `chapter4_real_only_v1` images and labels.
- Expert decisions completed: **0/25**.
- Every row remains `PENDING_EXPERT_REVIEW`; no YOLO or XML annotation was changed.
- Plates contain no prediction, AP, confidence, or XML/YOLO agreement score.

The review form is `EXPERT_REVIEW_TEMPLATE.csv`, the mechanical checks are in
`annotation_review_audit.csv`, and the images are in `annotation_review/`.

## Two boundary cases

| image | split | class | current YOLO box (xyxy, px) | historical XML box | observed difference | status |
|---|---|---|---|---|---|---|
| `img_1013_pre_part_1.jpg` | train | Big black spots | approximately `[1436,1010,1457,1023]` | `[1436,1010,1457,1024]` | bottom edge differs by about one pixel at the image boundary | PENDING_EXPERT_REVIEW |
| `img_9509_pre_part_0.jpg` | val | Big black spots | approximately `[4059,432,4095,467]` | `[4059,432,4096,467]` | right edge differs by about one pixel at the image boundary | PENDING_EXPERT_REVIEW |

These differences are consistent with a possible inclusive/exclusive coordinate or boundary-clipping convention, but that is an audit hypothesis—not an expert correctness decision. In particular, the validation GT was not altered.

## Expert procedure

Review View A and View B independently, inspect the supplied context, and fill only the expert columns. Use `AMBIGUOUS` or `NEED_MORE_CONTEXT` when appropriate. A corrected box is a recommendation only; it must not be applied until a separately versioned, reviewed annotation release is approved.

