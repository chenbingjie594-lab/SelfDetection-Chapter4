# FSPR calibration audit

Outer holdout GT was never used for fitting or threshold selection. The frozen E0 extractor was trained on the outer training set and therefore had seen inner-calibration images; this limitation is disclosed. Calibration JSON files include Brier score, ECE, counts and fitted parameters.
