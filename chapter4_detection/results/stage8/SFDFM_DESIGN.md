# SFDFM Design

SFDFM is inserted after original Neck P4 becomes available and refines the saved Neck P3 before Detect. P4/P5 construction remains on the original path.

Given aligned `F3` and projected `F4↑`, fixed Haar yields LH/HL/HH. A 1×1 detail encoder produces `H`; `D=|F3-F4↑|` and `S=F3·F4↑`; `G=sigmoid(psi([H,D,S]))`; and the output projection consumes `[F3,G·H,phi(D)]`.

The final projection is initialized as identity on F3 plus small (`σ=10⁻³`) branch weights. This is not gamma-zero routing: frequency, difference, gate, P4 projection and fusion parameters all receive non-zero gradients on the first backward pass. The three initialization audits show raw-output relative change around 0.7%, while every shared YOLO tensor is copied exactly.

The design follows the train-only evidence: directional edge statistics distinguish candidate boxes, broadband high frequency does not, and Neck P3 is the strongest shallow detection representation. No extra head, global attention, Q/K/V, loss or synthetic image is introduced.
