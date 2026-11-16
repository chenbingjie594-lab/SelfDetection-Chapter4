# Stage 7 Final Conclusion

Matched-Identity − Matched-E0 mean mAP50-95: `+0.0000`; paired seeds: `+0.0000, +0.0000, +0.0000`. Matched-Full − Matched-E0 mean: `+0.0026`; paired seeds: `-0.0203, +0.0061, +0.0221`.

## Directly established

- The historical E0/E3 initialization paths were not strictly matched.
- The new Identity control reproduces Matched-E0 mAP50-95 essentially exactly. The seed2026 E0 run required checkpoint resume, so exact trajectory identity is not claimed for that seed.
- Matched-Full improves mean mAP50-95 by only 0.0026, with mixed seed effects (one negative, two positive) and no independent test set. This is not robust evidence of a detection gain.
- Frozen-checkpoint bypass causes measurable but tiny, defect-concentrated changes; all 171 confidence≥0.25 detections remain one-to-one matched.

## Supported interpretation

Any NBR-DE effect is more plausible during training/representation optimization than as a strong inference-time background-enhancement mechanism. That explanation is supported, not proven causally.

## Decision

There is **not enough evidence to use NBR-DE as the formal Chapter-4 innovation**. Keep it as an exploratory result. A stronger direction should target a mechanism with reproducible paired-seed gains and a material inference-time effect, and must ultimately be checked on an independent real test set.
