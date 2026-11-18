# Stage 10 final summary

1. Black has modestly more training instances (93 vs 81), but both are extremely small after resize.
2. Joint small/low-contrast Black count is 33/93 under train-fixed median thresholds; see distributions, not averages alone.
3. No systemic format/conversion error is demonstrated; two XML mappings below IoU .99 require expert review and 35 images remain provenance-UNKNOWN.
4. Pixel-level perturbations materially affect tiny-box IoU; this is sensitivity, not proof of annotation error.
5. Persistent errors exist across models; counts and types are in the atlas.
6. CBCD OOF mAP50-95 is 0.3605 vs E0 0.3540, but AP75 is 0.2404 vs 0.2442; the bootstrap interval crosses zero and fold mAP gains are not unanimous. CBCD advantage is not stable evidence.
7. Evidence points primarily to tiny-target/data-coverage and precise-localization limits; annotation quality remains partly unresolved, not proven defective.
8. There is not yet sufficient evidence to add another module. Priorities are expert boundary review and localization-focused analysis; stop stacking CBCD variants.
