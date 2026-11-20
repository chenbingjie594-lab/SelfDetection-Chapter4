# Duplicate and source-group audit

- 9,059 exact SHA256 relation rows connect current images to copies in historical directories (`duplicate_and_source_relations.csv`).
- File SHA256 and the frozen decoded-pixel hash are both retained in the lineage table.
- Current train/val share neither source groups nor canonical IDs.
- Filename similarity and pHash similarity are never promoted to confirmed lineage by themselves.
- A crop/parent or near-duplicate relationship is accepted only with documentary evidence or local pixel/feature verification. Unverified relationships remain UNKNOWN.

The presence of current validation images in historical generation-study directories means those directories are not independent test sources. It is evidence of collection-level overlap, while the precise historical use (training reference, generation source, validation, or copied real control) must be read from a reliable use-role manifest.

