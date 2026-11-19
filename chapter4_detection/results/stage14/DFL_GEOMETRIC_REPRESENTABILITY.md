# DFL geometric representability

## Exact construction

For x-axis anchor coordinate (a_x), maximum distance (D_x), and GT edges ((x_1,x_2)), the closest reachable edges are

\[
\hat x_1=\min(\max(x_1,a_x-D_x),a_x),\qquad
\hat x_2=\max(\min(x_2,a_x+D_x),a_x).
\]

The y-axis is analogous. These four independently closest edges maximize intersection and minimize unnecessary predicted area, yielding the maximum IoU in the legal axis-aligned set. Here (D=14.99\times stride).

## Results

| Quantity | Unweighted | Loss-weighted |
|---|---:|---:|
| All positives with IoU-max < .50 | 9.69% | 0.42% |
| All positives with IoU-max < .75 | 32.12% | 5.92% |
| Outside positives with IoU-max < .50 | 15.56% | — |
| Outside positives with IoU-max < .75 | 51.59% | — |

At augmented-GT level, 28.54% had at least one assigned positive whose maximum IoU was below .50 and 60.12% had one below .75. For tiny GT groups these values were 51.33% and 91.19%, respectively. These are “at least one limited positive” rates; they do not mean the GT lacked all high-quality alternatives.

The count is substantial, but TAL sharply downweights many limited positives: the weighted shares (0.42% below .50 and 5.92% below .75) are much smaller than their count shares. Therefore the geometry can plausibly contribute to small-object localization difficulty, but the trace does not support treating every outside positive as equally harmful.

Evidence: `analysis/geometry_summary.json`, `analysis/geometry_by_augmented_gt.csv`, and `evidence/positive_geometry_instances.csv`.
