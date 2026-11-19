# Geometric-supervision dynamics

## Protocol and non-invasiveness

Fold0, seed42, imgsz1536, batch1, 150 epochs, matched Stage10 initialization and training options. Geometry was recomputed under `torch.no_grad()` only after the untouched parent loss returned. Detection-level boundaries used actual feature shapes, not an inferred square grid.

The monitored and Stage10 baseline runs are exactly identical:

- 499/499 checkpoint tensors equal; maximum absolute difference 0.
- 150×14 non-time CSV values equal; maximum absolute difference 0.
- Same initialization SHA256: `8173afed51dc8367a0c22643a02d60184a646d551536196d03bcb037cedc76f9`.

This end-to-end equality is stronger than a sampled loss check: optimizer updates and final state are unchanged.

| Epoch | Positives | Outside count | Outside weight |
|---:|---:|---:|---:|
| 1 | 760 | 55.66% | 27.44% |
| 3 | 1,062 | 61.86% | 39.15% |
| 10 | 1,050 | 62.86% | 48.32% |
| 25 | 959 | 66.74% | 41.02% |
| 50 | 1,009 | 63.43% | 36.85% |
| 100 | 1,036 | 66.12% | 37.39% |
| 150 | 749 | 56.07% | 30.08% |

Outside positives persist throughout training rather than being only an initialization artifact. Their weights vary and remain below their count share.

## Stage13 scale audit

Stage14 derives P3/P4/P5 boundaries from `preds['feats'][i].shape[2:4]` on every call. The monitored training input is square (1536×1536), so Stage13's square-grid assumption did not change its Fold0 training-scale results. That assumption must not be reused for rectangular inference; Stage14 code removes it.
