# Independent Test Audit

## Decision

**UNKNOWN — no candidate qualifies as a formal independent test set.**

The 46-image validation set has already guided model design and is not an independent test set. Historical directories named `test` or `eval` were audited conservatively; a directory name is not provenance.

| Candidate | Images | Status | Exact overlaps | Known source-name overlaps | Near candidates (dHash≤8) |
|---|---:|---|---:|---:|---:|
| `/mnt/sda/cbj/dataset/Plastic_Bomo/Plastic_Bomo_mvtec_png/Plastic_Bomo/test` | 261 | UNKNOWN | 0 | 0 | 22 |
| `/mnt/sda/cbj/dataset/Plastic_Bomo/Plastic_Bomo_mvtec_ready/Plastic_Bomo/test` | 261 | UNKNOWN | 0 | 0 | 22 |
| `/mnt/sda/cbj/dataset/Plastic_Bomo/Plastic_Bomo_split70_seed42/eval` | 514 | UNKNOWN | 0 | 0 | 28 |
| `/mnt/sda/cbj/dataset/Plastic_Bomo/Plastic_Bomo_split70_seed42/eval/Plastic_Bomo/test` | 47 | UNKNOWN | 0 | 0 | 0 |
| `/mnt/sda/cbj/dataset/Plastic_Bomo/Plastic_Bomo_split70_seed42/train/Plastic_Bomo/test` | 112 | UNKNOWN | 0 | 0 | 0 |

The perceptual-hash screen found near-duplicate candidates in several legacy collections, but missing source provenance prevents deciding whether every pair is the same physical source. They remain `UNKNOWN`, not independent. Candidates with proven overlap would be `FAIL_OVERLAP`; all candidates here lack complete source-group provenance and proof of non-use during historical development. No formal test evaluation was run, and the frozen train/val split was not changed.
