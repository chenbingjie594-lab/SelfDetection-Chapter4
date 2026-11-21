# CIR efficiency

- A: 74340 parameters, 298.2 KiB, CPU local inference 117.1 ms/original image, mean observed CPU training 24.8 s/fold.
- B: 74340 parameters, 298.2 KiB, CPU local inference 92.3 ms/original image, mean observed CPU training 130.6 s/fold.
- C: 90724 parameters, 362.2 KiB, CPU local inference 70.2 ms/original image, mean observed CPU training 140.4 s/fold.
- C-noDiff: 90724 parameters, 362.2 KiB, CPU local inference 78.1 ms/original image, mean observed CPU training 139.2 s/fold.

Parameter counts and checkpoint sizes are reliable. Wall-clock values are approximate: system load was uncontrolled and the initial fold-0 B/C/C-noDiff jobs overlapped, so timings must not be used to rank variant efficiency.
