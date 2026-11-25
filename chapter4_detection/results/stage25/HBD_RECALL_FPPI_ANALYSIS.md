# Recall–FPPI analysis

FPPI counts only predictions whose maximum IoU to any-class GT is below 0.10. Thresholds were frozen before comparison: 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.4, 0.5, 0.75. Fixed-FPPI summaries are conservative maxima over this frozen grid; they are diagnostic, not a tuned deployment threshold.

- E0: recall@FPPI≤0.1=0.8621; conf=.25 recall=0.8276, FPPI=0.0435.
- M2: recall@FPPI≤0.1=0.8506; conf=.25 recall=0.8736, FPPI=0.1739.
- M3: recall@FPPI≤0.1=0.8103; conf=.25 recall=0.8793, FPPI=0.2174.
- H1: recall@FPPI≤0.1=0.4483; conf=.25 recall=0.8563, FPPI=0.4348.
- H2: recall@FPPI≤0.1=0.4425; conf=.25 recall=0.8333, FPPI=0.2826.
- H3: recall@FPPI≤0.1=0.4598; conf=.25 recall=0.8448, FPPI=0.2536.
- H4: recall@FPPI≤0.1=0.4540; conf=.25 recall=0.8506, FPPI=0.5797.
