# Stage19 decision

## `FAIL_MECHANISM_RECHECK`

The 1536 audit is complete, hard-background definitions are fixed and reviewable, and r has a nontrivial trained-feature range. Nevertheless, the necessary selective-effect gates fail: high-D and high-class backgrounds are not suppressed more than real Black defects; low-contrast Black receives comparable suppression; and matched-D comparisons can favor backgrounds over GT. RGLCD therefore does not justify formal three-fold training.

No alpha, ring, epsilon, gamma initialization, projection or formula is changed. No holdout performance was read. No Stage19 formal freeze, CV metrics or trained interventions are generated. This closes the RGLCD route rather than starting v2/v3 searches. CUDA remains unavailable, but the decision is a mechanism failure—not `INCONCLUSIVE_ENVIRONMENT_BLOCKED`, because CPU completed the required 1536 audit.

