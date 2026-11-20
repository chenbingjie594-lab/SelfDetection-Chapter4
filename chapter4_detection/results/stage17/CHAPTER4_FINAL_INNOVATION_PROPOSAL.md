# Final innovation proposal

## Candidate ranking

1. **Reliability-Gated Local Contrast Discriminator (RGLCD)** — recommended. Targets periodic-background false activation while preserving regression geometry. Strongest relevance; mixed but direct support from SFDFM/CBCD/NBR-DE; low cost and testable.
2. **Subpixel Boundary Sampling for P3 regression.** Samples four predicted boundary neighborhoods and adds a zero-initialized DFL-logit residual. It targets 12–14% width/height error, but PCRF and native-scale failures make risk high.
3. **Quality-conditioned classification score.** Predicts localization quality and calibrates class scores. Local ranking errors exist, but candidate generation—not ranking—is the largest failure class, so this is lower priority.

## Recommended method: RGLCD

Insert only before the P3 classification tower; P3 regression and P4/P5 are unchanged. For feature `F`, compute a depthwise local center response `C(F)` and an annular background estimate `B(F)`. Define standardized contrast

`D=(C-B)/(sqrt(Var_ring(F)+eps))` and reliability `r=exp(-alpha*Var_ring(F))`.

A zero-initialized projection produces `F_cls = F + gamma * r * phi(D)`, with scalar `gamma=0` initially. The identity start preserves baseline outputs; low-stationarity rings downweight unreliable references. Unlike NBR-DE, this uses a spatially aligned local annulus rather than global learned reference similarity. Unlike CBCD, it normalizes the defect residual by measured background variability. Unlike SFDFM, it does not inject broad frequency energy or alter regression geometry. This is a proposed contribution, not an established novelty claim; a literature review remains necessary before thesis wording.

Expected overhead is measured, not guessed: parameter delta from model summaries, GFLOPs at 1536, synchronized latency and peak CUDA memory. Depthwise fixed/local statistics plus one small projection should be modest, but no numeric claim is made before implementation.

## Minimal frozen experiment

Use Stage10 source-group folds, matched 2-class E0 initialization, seed42, imgsz1536, 150 epochs, `last.pt`, identical optimizer/augmentation/evaluator. Compare E0 versus RGLCD on all three folds—never tune after Fold0. Primary gates: merged OOF mAP50-95 and AP75 must both be non-negative; no fold may show catastrophic AP75 loss (>0.03); background-like FP must not rise >10%. Report paired source-group bootstrap and treat crossing-zero intervals as uncertainty.

If the frozen pilot passes, run seeds 3407/2026 unchanged. If it fails, stop this mechanism rather than tuning on the same OOF.

## Ablations

- Full RGLCD.
- Remove reliability (`r=1`) to test whether ring trust matters.
- Remove standardization (`D=C-B`) to test whether variance-normalized contrast matters.

These are two independent mechanisms. The identity `gamma=0` is an implementation/control check, not an extra performance ablation.

## Failure modes and exit

Periodic background may also be locally stationary, annuli may contain the defect or another defect, and 174 training GTs may overfit the gate. Exit if the primary gate fails, bootstrap uncertainty plus fold signs show no coherent effect, AP75 degrades, or the learned residual is inference-negligible as in NBR-DE. Do not launch another background module without new evidence.

