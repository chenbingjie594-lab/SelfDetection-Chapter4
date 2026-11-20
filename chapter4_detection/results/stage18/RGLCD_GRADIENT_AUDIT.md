# RGLCD gradient audit

Using a real Fold0 training image, label and the actual YOLO detection loss on CPU:

| step | loss | gamma before→after | gamma grad | phi grad | finite |
|---:|---:|---:|---:|---:|---|
| 0 | 30.1283 | 0→-.001000 | 1.8075 | 0 | yes |
| 1 | 32.2280 | -.001000→-.001998 | 2.3424 | .02712 | yes |
| 2 | 19.0268 | -.001998→-.002885 | 7.1125 | .22344 | yes |

The expected staged routing is confirmed: gamma learns at identity initialization; phi is zero-gradient only while gamma is zero and receives a finite nonzero gradient afterward. P3 classification and regression towers also received finite gradients (the third-step regression gradient happened to be zero for that batch/loss state and is not presented as disconnection). No NaN/Inf occurred.

