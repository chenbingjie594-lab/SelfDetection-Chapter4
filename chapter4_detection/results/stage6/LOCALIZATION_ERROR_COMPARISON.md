# Localization and Error Comparison

The analysis uses class-aware one-to-one matching. IoU rates use all 58 GT instances; center/width/height errors use IoU≥0.50 matches only. Fixed-confidence FP categories separate background, class confusion, insufficient localization, and duplicates.

| Model | Seed | IoU≥.50 | IoU≥.75 | IoU≥.90 | Center px | Width rel. error | Height rel. error |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | 42 | 0.8448 | 0.4655 | 0.0517 | 1.9822 | 0.1093 | 0.1206 |
| E0 | 3407 | 0.8448 | 0.4483 | 0.0345 | 2.0700 | 0.1111 | 0.1281 |
| E0 | 2026 | 0.8103 | 0.4138 | 0.0517 | 1.9920 | 0.1047 | 0.1405 |
| E3 | 42 | 0.8448 | 0.4828 | 0.0517 | 1.9958 | 0.1000 | 0.1167 |
| E3 | 3407 | 0.8276 | 0.4828 | 0.0862 | 2.0536 | 0.1046 | 0.1104 |
| E3 | 2026 | 0.8103 | 0.5345 | 0.0862 | 1.8509 | 0.0889 | 0.1237 |
| A1 | 42 | 0.8276 | 0.4310 | 0.1034 | 2.2227 | 0.1247 | 0.1274 |
| A1 | 3407 | 0.8276 | 0.5000 | 0.0517 | 2.0965 | 0.1141 | 0.1166 |
| A1 | 2026 | 0.8276 | 0.5345 | 0.0517 | 1.9906 | 0.1002 | 0.1189 |

At IoU≥0.75, E3 exceeds the corresponding E0 rate in all three seeds, but the bootstrap AP75 interval remains wide and crosses zero. Improvements are distributed across multiple matched targets rather than one target, yet the 58-instance sample is too small to exclude dominance by a small subset. A1's higher mean AP75 is not statistically secure. The saved per-instance CSV and five side-by-side figures provide the auditable target-level evidence.
