# 标注人工复核状态

Stage10 的 25 个优先样本及图版继续沿用 `../stage10/MANUAL_REVIEW_TABLE.csv` 和 `../stage10/annotation_review/`。所有专家结论仍为空，统一状态为 `PENDING_EXPERT_REVIEW`，本阶段未修改任何 GT。

优先补充核查两张 XML/YOLO 匹配低于 0.99 的图像：

| 图像 | split | 类别 | YOLO xyxy | 原图宽高 | imgsz1536宽高 | XML最优IoU | 状态 |
|---|---|---|---|---:|---:|---:|---|
| img_1013_pre_part_1.jpg | train | Big black spots | [1435.998,1010.000,1456.998,1023.000] | 21.00x13.00 | 7.88x4.87 | 0.9284 | PENDING_EXPERT_REVIEW |
| img_9509_pre_part_0.jpg | val | Big black spots | [4058.999,432.000,4094.999,467.000] | 36.00x35.00 | 13.50x13.13 | 0.9729 | PENDING_EXPERT_REVIEW |

两者均能追溯到原始 XML，但差异原因尚未由专家确认。第一张位于图像下边界，几何扰动尤其敏感；第二张在 OOF 中多次属于 IoU50 达到但 IoU75 未达到的样本。该不确定性不阻塞技术诊断，但限制对标注质量和精细定位上限的正式结论。
