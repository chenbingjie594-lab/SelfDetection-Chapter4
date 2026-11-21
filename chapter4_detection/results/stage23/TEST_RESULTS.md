# Stage23 test results

- Stage22 + Stage23 unit tests: **6 passed in 1.62 s**.
- Python syntax compilation: **passed**.
- Corrected evaluator tests cover best-candidate ID switching and preservation by an alternate same-class candidate.
- Gate tests cover g=0, g=0.5 and g=1 geometry, effective gradients and legal residual decoding.
- The three cross-fit target audits each report zero overlap with their outer holdout.
- The final prediction invariance audit requires identical candidate count, class order and confidence scores for every model.
