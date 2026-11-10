# chapter4_real_only_v1 audit report

- Frozen path: `/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/`
- Audit status: `PASS`
- Protocol SHA256: `48054c137c8e093957c57eef6d761750cfac6f7b437e089cebe42053fb4b62ab`
- Real/synthetic images: 184/0
- Train: 138 images, 174 instances (Flash 81, Black 93), 137 source groups
- Validation: 46 images, 58 instances (Flash 27, Black 31), 44 source groups
- Cross-split source groups: 0
- Cross-split file SHA256 duplicates: 0
- Cross-split decoded-pixel SHA256 duplicates: 0
- Unknown provenance records: 0
- Image size: all 4096×1024
- Label format: two-class YOLO HBB, all rows valid

Provenance levels: 149 original XML confirmations, 25 exact historical black-image
reconstructions, and 10 deterministic historical black-export IDs with lower evidence grade.
The two known sibling pairs (`img_1020_pre_part_0/1`, `img_9509_pre_part_0/1`) are each
contained in one split. See `NEAR_DUPLICATE_REVIEW.md` for the separate texture-similarity review.
