# RGLCD group-CV status

Formal Fold0/1/2 training was **not run**. The decision was made before reading any outer-holdout metric because Fold0-train mechanism diagnosis showed the frozen `alpha=1` reliability term was nearly constant (`r≈0.989`) and did not distinguish GT from sampled background. The NVIDIA driver was also unavailable.

No RGLCD Precision, Recall, mAP, AP75, class AP, candidate coverage, false-positive, duplicate, per-fold or bootstrap values exist. Stage10 E0 numbers are not copied into empty RGLCD cells, and no result is fabricated. Decision: `FAIL_PRETRAIN_MECHANISM_GATE`.

