# Near-duplicate review for chapter4_real_only_v1

The audit's 256-bit dHash threshold produced 12 diagnostic candidates. Four cross-split pairs were
reviewed at 512×128 grayscale resolution. They have distinct original XML `filename/path` source
identities and the following pixel comparisons:

| Left | Right | dHash | Pearson r | MAE | SSIM |
|---|---|---:|---:|---:|---:|
| img_16871_pre_part_0 | img_7241_pre_part_0 | 2 | 0.9658 | 8.00 | 0.7447 |
| img_7241_pre_part_0 | img_8935_pre_part_0 | 2 | 0.9772 | 5.59 | 0.7485 |
| img_7241_pre_part_0 | img_4350_pre_part_0 | 8 | 0.9576 | 8.40 | 0.5862 |
| img_3674_pre_part_0 | img_4350_pre_part_0 | 7 | 0.9717 | 9.09 | 0.5840 |

Disposition: not pixel near-duplicates. The low dHash distances and high global correlations are
caused by the shared, low-frequency strip texture; SSIM and pixel residuals show materially different
texture fields. These pairs are not merged without contrary provenance evidence. This review does not
claim that every distinct capture necessarily came from a different physical production batch; that
batch-level identity is unavailable and is outside the historical annotation provenance.
