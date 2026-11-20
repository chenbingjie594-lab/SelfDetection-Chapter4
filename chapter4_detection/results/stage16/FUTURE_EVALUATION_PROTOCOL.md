# Future evaluation protocol

## A. Exploratory evidence

Stage10–15 results, three-fold OOF results, and the repeatedly used 46-image validation set are exploratory/development evidence. They can support within-protocol mechanism comparisons, sensitivity observations, and hypothesis formation. They cannot support a claim of performance on a never-seen external population.

## B. Development evaluation

Use source-group-isolated folds and report all prespecified folds/seeds, effect sizes and uncertainty. Keep annotation-review status visible. Freeze thresholds, metrics, checkpoint policy and exclusion rules before evaluating. Do not repeatedly reuse OOF outcomes for unconstrained architecture search.

## C. Independent final test

Acquire and lock a traceable new production batch after method selection. Run hash, decoded-pixel, canonical-ID, source-group, near-duplicate and crop-parent checks against all development/generation data. Have labels reviewed without model predictions. Seal a manifest and hashes, then evaluate exactly once without tuning.

Until such a set exists, the thesis must state `NO_VERIFIED_INDEPENDENT_TEST_SET` and limit generalization claims accordingly. No new significance or generalization conclusion is inferred by Stage16.

