# Annotation review handoff

All 27 items are present in `expert_review/` with stable IDs `S17-001`–`S17-027`. The first 25 preserve Stage16 content; S17-026/027 add `img_1013_pre_part_1` and `img_9509_pre_part_0`. Views show YOLO and XML separately, at original-pixel-derived crops, with no predictions or model metrics.

`STAGE17_EXPERT_REVIEW.csv` is deliberately unfilled: all rows are `PENDING_EXPERT_REVIEW`. After expert completion run `stage17_import_expert_review.py`. It validates IDs, manifest membership, status, class and box bounds, then writes proposals only. It never edits labels. The current import result is `PENDING_HUMAN_REVIEW`.

The expert must still determine object validity, class correctness, boundary quality and any corrected box. The one-pixel image-edge differences in S17-026/027 must not be automatically treated as errors.

