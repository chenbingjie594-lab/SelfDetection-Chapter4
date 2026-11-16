# Training Dynamics Report

This is a fresh seed42 diagnostic run, not a reconstructed historical trajectory. It records 300 successful optimizer updates and 12 GradScaler-skipped attempts.

- gamma: start `-0.000112743`, final `-0.0088152`, maximum absolute `0.0088152`.
- injection ratio rho: mean `0.00255937`, maximum `0.00857815`, final `0.00853711`.
- gate mean: mean `0.47141`, final `0.461822`.

The CSV contains Q/K/V, difference, gate, adjacent-layer gradients and box/cls/dfl losses for every optimizer attempt. Validation values from this diagnostic run were not used for tuning.
