# RGLCD frozen design

For P3 feature `F`, a fixed 7×7 ring excludes the central 3×3. Border counts include only valid pixels. Per-channel float32 statistics are `B=ΣMF/N`, `V=max(ΣMF²/N-B²,0)`, `D=(F-B)/sqrt(V+1e-4)`, and `r=exp(-V)`. Classification receives `F_cls=F+gamma*r*phi(D)`; regression receives `F`. `phi` is a normally initialized 1×1 projection and `gamma=0`.

P4/P5, Detect towers, strides, DFL, decoding, NMS and loss are unchanged. The implementation does not claim that training-time regression is causally unchanged because the backbone/neck remains shared.

