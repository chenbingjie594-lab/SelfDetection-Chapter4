# Stage21 final summary

1. Frozen-prediction AP75 is materially sensitive to subpixel-to-two-pixel GT boundary perturbations; tiny targets are generally more sensitive.
2. This demonstrates evaluation sensitivity, not annotation error. Expert review remains pending and no label was changed.
3. Model width/height error and one-to-two-pixel relative boundary changes are of similar order for many tiny GT.
4. Low-dimensional deployable geometry/DFL features show only weak signed-residual predictability. Ridge improves residual MAE by 4.52% versus fold-local median but worsens high-quality coverage.
5. There is no evidence that Flash and Black require separate scale regressors.
6. SCR fails the frozen mechanism gate, was not implemented, and should not enter formal training. The next step should be expert boundary review or new independent real data, not another regression branch.
