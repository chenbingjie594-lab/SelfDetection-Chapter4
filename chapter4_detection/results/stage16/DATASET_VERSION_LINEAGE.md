# Dataset version lineage

The audit scanned only the two authorized roots and found 17,594 image files in 71 directory-defined versions. `dataset_version_inventory.csv` records counts/splits/classes and `all_image_file_hashes.csv` records every file SHA256. Directory-defined versions are an inventory unit, not proof of provenance.

## Frozen detection protocol

- `chapter4_real_only_v1`: 138 train / 46 validation.
- Train/validation source-group intersection: **0**.
- Train/validation canonical-image-ID intersection: **0**.
- The frozen data were not modified.

## Historical relationships

Exact copies of current images occur throughout DWBG, GLRR, bootstrap-guard, real-only and generation-study directories. Exact-match current-image coverage includes:

| historical family | current images found | train | val |
|---|---:|---:|---:|
| DWBG-named paths | 184 | 138 | 46 |
| GLRR-named paths | 184 | 138 | 46 |
| bootstrap-named paths | 184 | 138 | 46 |
| Stage14-named paths | 138 | 99 | 39 |
| Stage15-named paths | 138 | 99 | 39 |
| Stage17-named paths | 138 | 99 | 39 |
| `PlasticBomo_detect2` | 149 | 113 | 36 |

Therefore these historical collections cannot be treated as new independent evaluation data. The Stage14–17 intersections prove file reuse in those collections; they do not, without a use-role manifest, prove that every intersecting file was fed to every generator or detector.

No uniquely identified, auditable manifest explicitly defining the stated historical **136/48** protocol was found under the two authorized roots. Its exact membership and role therefore remain UNKNOWN; it is not substituted by a similarly sized directory. The concrete file intersections above are retained as evidence.

OOF is a regrouping of development images and E2 slices are derived crops; neither creates independent sources. Generated/modified variants are not eligible as real test images.

