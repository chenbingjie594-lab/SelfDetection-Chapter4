# Reference reliability validation

On trained-E0/1536, r correlates negatively with independent ring directional/structure inconsistency, as intended: correlations are approximately G1 `-.915/-.934`, G2 `-.803/-.872`, G3 `-.831/-.847`, Flash `-.762/-.719`, and Black `-.844/-.726`. Thus r does reflect local instability, not only by tautological comparison to V.

However this behavior is not selectively favorable. Mean r is .9497 for G3, .9540 for Flash, .9419 for Black and .9442 for low-contrast Black. Low-contrast Black is suppressed 5.40%, comparable to G3's 4.89% and less favorably than G2's 3.02%. At matched `2≤|D|<4`, mean r/suppression are: Flash .9325/5.15%, Black .9063/8.80%, low-contrast Black .9463/5.92%, G2 .9661/3.45%, G3 .9527/4.83%. Difficult backgrounds are not preferentially suppressed relative to real defects.

The dynamic range after E0 training is nonzero (G3 mean-channel r range .8527–.9904), whereas initialization remains concentrated near one. This establishes sensitivity to reference instability but not useful class selectivity.

