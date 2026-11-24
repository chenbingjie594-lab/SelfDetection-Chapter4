# MBRD efficiency

- M1: 16449 trainable adapter parameters; mean observed CPU full-image inference 0.475 s.
- M2: 16449 trainable adapter parameters; mean observed CPU full-image inference 0.507 s.
- M3: 16449 trainable adapter parameters; mean observed CPU full-image inference 0.435 s.

CPU wall times are approximate and were measured under uncontrolled system load. M3 additionally performs vectorized P3-to-32-prototype similarity and top-4 retrieval.
