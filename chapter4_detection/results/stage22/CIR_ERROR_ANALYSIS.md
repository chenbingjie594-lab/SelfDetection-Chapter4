# CIR error analysis

- A: mean matched-candidate IoU 0.7184→0.7119; coverage@.75 0.4598→0.4253; 15 gains and 21 losses.
- B: mean matched-candidate IoU 0.7184→0.7160; coverage@.75 0.4598→0.4368; 13 gains and 17 losses.
- C: mean matched-candidate IoU 0.7184→0.7182; coverage@.75 0.4598→0.4310; 14 gains and 19 losses.
- C-noDiff: mean matched-candidate IoU 0.7184→0.7136; coverage@.75 0.4598→0.4253; 14 gains and 20 losses.

Candidate count, score and class are unchanged, so background-candidate count cannot increase. Box movement can still alter duplicate overlap; per-fold audits are retained. Both successful and harmful corrections are included.
