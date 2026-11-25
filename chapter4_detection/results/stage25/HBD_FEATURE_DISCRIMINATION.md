# HBD feature-discrimination audit

Training projection heads separate the tiny cached positive/B sets strongly, and the HBD loss falls near zero. This is not evidence of holdout improvement: only 12–15 B samples exist per fold, so the constraint is easy to memorize.

At holdout output level, the candidate-dependent ROC-AUC/PR-AUC values are:

| Model | Positive score mean | Background score mean | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| E0 | .6081 | .0223 | .9685 | .9731 |
| M2 | .6804 | .0162 | .9725 | .9521 |
| M3 | .6788 | .0170 | .9617 | .9430 |
| H1 | .6670 | .0179 | .9496 | .8809 |
| H2 | .6591 | .0173 | .9631 | .9126 |
| H3 | .6599 | .0148 | .9588 | .9081 |
| H4 | .6801 | .0191 | .9682 | .9038 |

Negatives are each model's post-NMS background-style candidate set, so its size changes with the model; these AUCs are diagnostic rather than a common-candidate causal comparison. H3 lowers the mean negative score but also lowers the positive mean and greatly increases the number of low-score background candidates. The FROC result confirms this is not improved discrimination at a fixed false-positive budget.
