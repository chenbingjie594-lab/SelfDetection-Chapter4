# 定位质量排序可行性

## 与已有方法的边界

- [IoU-Net](https://arxiv.org/abs/1807.11590) 增加定位置信度预测并用其改善 NMS/框优化。
- [Generalized Focal Loss](https://arxiv.org/abs/2006.04388) 将 IoU 质量并入类别表示，以 QFL 学习联合分数，同时用 DFL 表示边界分布。
- [VarifocalNet](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.html) 直接学习 IoU-aware classification score，并以 Varifocal Loss 优化。
- [GFLv2](https://arxiv.org/abs/2011.12885) 从四边分布统计构建轻量 DGQP 定位质量估计器。

这些都是成熟的 IoU-aware/quality-aware 设计，不能换名作为创新点；采用它们通常需要额外质量分支、分类监督修改或两者之一。

## 本数据判断

分类分数与 IoU 的 source-group Spearman 为约 .71，明显高于任一 DFL 单特征（绝对值≤.27）。高 IoU 失败最大类是候选生成不足，极小组尤其明显。提高 NMS 阈值虽然提高高质量框保留数，却增加背景和重复预测并降低总体 AP。

因此任务六四个门槛未同时满足：存在局部失配，但“损失主要由排序造成”和“可推理组合特征已有足够预测证据”不成立；138张小数据也不足以抵消候选分布偏差风险。按预注册规则不拟合 MLP、不执行R1，也不产生 `QUALITY_RERANKING_EXPERIMENT.md`。这避免用当前 OOF 标签反复选特征/超参数。未来若有独立来源数据，可预注册比较原始 YOLO、QFL/VFL/GFLv2，而不是先命名新检测头。
