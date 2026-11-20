# Targeted test-candidate triage

Stage16's 681 `LABELS_UNVERIFIED` rows were read directly; the 17,594-image scan was not repeated. They form 10 directory groups, dominated by MVTec-ready JPEG, lossless PNG copies, and a historical seed42 70/30 split.

Defect-class directories have paired segmentation masks in the MVTec `ground_truth` tree, which is useful annotation evidence but is not verified human HBB provenance. `01_Flash_point` and `02_Big_black_spots` are image-level directory labels and were not converted into fabricated boxes. Normal groups have no object boxes, but their acquisition independence is also unproven. The split70 copies have explicit historical development use.

No group has a verified new acquisition batch, a source-group mapping, complete trusted HBB provenance, and proof of non-use during development. Result: **`NO_VERIFIED_INDEPENDENT_TEST_SET`**. The mask-bearing non-split70 groups are medium priority for a human provenance review; they are not eligible test data yet.

