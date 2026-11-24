# MBRD failure decomposition

- M1: {'E_correctly_retained': 67, 'B_score_misranking': 24, 'A_generation_insufficient': 83}; Flash score-misranking=14, Black score-misranking=10.
- M2: {'E_correctly_retained': 69, 'B_score_misranking': 22, 'A_generation_insufficient': 83}; Flash score-misranking=14, Black score-misranking=8.
- M3: {'E_correctly_retained': 67, 'B_score_misranking': 23, 'A_generation_insufficient': 84}; Flash score-misranking=16, Black score-misranking=7.
- E0: {'C_nms_suppression': 21, 'E_correctly_retained': 48, 'B_score_misranking': 23, 'A_generation_insufficient': 82}; Flash score-misranking=17, Black score-misranking=6.

The regression tower is frozen, so MBRD cannot create new box geometry; changes arise from P3 classification scores and subsequent NMS/max-det retention. The compact diagnostic exports every anchor-class pair above 0.001, but it does not trace the exact internal NMS suppression decision and therefore is not interchangeable with the full Stage12 trace.
