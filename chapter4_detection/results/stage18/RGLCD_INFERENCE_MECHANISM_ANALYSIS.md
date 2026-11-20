# Inference mechanism analysis

At initialization, gamma=0 makes module-on and module-off outputs exactly identical. A three-step real-loss audit proves the branch can begin learning, but it is not a trained detector.

Because formal training was stopped before CV, there is no legitimate trained RGLCD checkpoint on which to perform the requested on/off causal intervention. Therefore changes in classification logits, final predictions, mAP50-95, AP75 and background FP are **not available**. Nonzero gamma after three diagnostic optimizer steps is not treated as efficacy evidence.

