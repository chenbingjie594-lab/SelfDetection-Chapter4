# MBRD training results

Nine adapters completed 40 epochs with AdamW and last-checkpoint selection. Backbone, neck, box tower and original class tower parameters are frozen. Cached targets are the exact fold-specific E0 TAL assignment; this freezes assignment rather than allowing MBRD scores to alter TAL online and is a disclosed difference from full YOLO training.

- Fold 0 M1: γ=-0.23953, loss=0.008214, P3 logit |Δ|=0.823546.
- Fold 0 M2: γ=-0.23686, loss=0.008215, P3 logit |Δ|=0.900243.
- Fold 0 M3: γ=-0.23675, loss=0.008195, P3 logit |Δ|=0.860333.
- Fold 1 M1: γ=-0.22130, loss=0.009099, P3 logit |Δ|=1.071509.
- Fold 1 M2: γ=-0.22898, loss=0.009086, P3 logit |Δ|=0.953059.
- Fold 1 M3: γ=-0.23083, loss=0.009048, P3 logit |Δ|=1.039941.
- Fold 2 M1: γ=0.25492, loss=0.008575, P3 logit |Δ|=0.866640.
- Fold 2 M2: γ=0.25961, loss=0.008542, P3 logit |Δ|=0.847153.
- Fold 2 M3: γ=0.25915, loss=0.008514, P3 logit |Δ|=0.867113.
