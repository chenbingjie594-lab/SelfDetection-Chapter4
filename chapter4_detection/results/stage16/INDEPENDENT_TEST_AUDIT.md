# Independent test audit

## Verdict

`NO_VERIFIED_INDEPENDENT_TEST_SET`

The scan found 777 images under directories named `test`. None passed all admission criteria:

| status | images |
|---|---:|
| LABELS_UNVERIFIED | 681 |
| PROVENANCE_UNKNOWN | 75 |
| OVERLAP_WITH_TRAIN | 16 |
| OVERLAP_WITH_VAL | 5 |
| ELIGIBLE_INDEPENDENT | 0 |

The detailed reasons and exact hashes are in `independent_test_candidates.csv`. A directory called `test`, a different filename, or a copy in another version is not evidence of an unseen acquisition batch.

Required new evidence: a new real production/acquisition batch; capture date and batch identity; original unmodified images; complete expert detection labels; canonical IDs/source groups; non-overlap evidence against all development and generation-source data; and confirmation that the batch was never used for architecture or hyperparameter selection. Evaluation should happen once after freezing the method.

