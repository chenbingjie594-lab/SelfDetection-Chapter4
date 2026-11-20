# Annotation boundary sensitivity

Frozen Stage10 E0 OOF predictions were never changed; only temporary GT copies were perturbed. Ten deterministic random seeds were used per non-baseline condition. Every run retained 174 valid boxes and unchanged classes/counts.

Baseline: AP50 0.8061, AP75 0.2442, mAP50-95 0.3540, IoU≥.75 coverage 0.4598, tiny coverage 0.3269.

| amplitude | center ΔAP75 | size ΔAP75 | independent-edge ΔAP75 | center tiny coverage Δ | independent tiny coverage Δ |
|---:|---:|---:|---:|---:|---:|
| 0.5 px | -0.0141 | -0.0092 | -0.0173 | -0.0288 | -0.0173 |
| 1 px | -0.0412 | -0.0114 | -0.0252 | -0.0423 | -0.0231 |
| 2 px | -0.1008 | -0.0741 | -0.0917 | -0.1173 | -0.0942 |

At ±2 px, center and independent-edge perturbations reduce AP75 by 0.1008 and 0.0917 on average. Tiny coverage falls more strongly than overall coverage for these conditions. Random perturbations may occasionally improve individual matches; this experiment measures evaluation sensitivity and is not evidence of annotation error or model improvement.

See `annotation_perturbation_results.csv` and `annotation_boundary_sensitivity.png`.
