# E0 YOLO11s seed42

- Status: completed normally with early stopping
- Data protocol: `chapter4_real_only_v1`
- Data protocol SHA256: `48054c137c8e093957c57eef6d761750cfac6f7b437e089cebe42053fb4b62ab`
- Training: 138 images, 150 maximum epochs, patience 40, batch 1, imgsz 1536
- Actual optimizer selected by `optimizer=auto`: AdamW, initial lr 0.001667, momentum 0.9
- Completed epochs: 76
- Best epoch: 36 (`results.csv` mAP50-95 0.41908)
- Checkpoint rule: `best.pt` selected by full-image validation mAP50-95
- `best.pt` SHA256: `873d51672475ffc081817ef3740aa81ad2d9940a1be02f1d10750f14d38b0399`
- `last.pt` SHA256: `4af0c16e6b951f5b131b385be0a086612facd33c39458896af17ffc1a792ef53`

## Fixed reevaluation of best.pt

| Metric | Value |
|---|---:|
| Precision | 0.8343579 |
| Recall | 0.7532855 |
| mAP50 | 0.8375577 |
| mAP50-95 | 0.4170290 |
| AP75 | 0.3499543 |
| Flash recall | 0.9259259 |
| Flash AP50 | 0.8955427 |
| Flash AP50-95 | 0.4137815 |
| Black recall | 0.5806452 |
| Black AP50 | 0.7795727 |
| Black AP50-95 | 0.4202764 |
| Parameters | 9,428,566 |
| GFLOPs at 1536 | 128.1803 |
| Inference | 8.4923 ms/image |
| Total preprocess + inference + postprocess | 10.4213 ms/image |
| Peak allocated GPU memory during reevaluation | 333,988,352 bytes |

The trainer's terminal validation and the separate fixed evaluator use different validation batch
execution paths and therefore produced slightly different values. The formal table uses the separate
fixed evaluator (`e0_seed42_metrics.json`) consistently intended for cross-model comparison.

This is one seed only. No mean, standard deviation, or E0/E1/E2 causal conclusion is reported yet.
