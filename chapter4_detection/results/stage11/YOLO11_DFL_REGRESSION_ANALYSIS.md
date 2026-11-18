# YOLO11 DFL 回归过程审计

环境为 Ultralytics 8.4.145。YOLO11s Detect 的 P3/P4/P5 stride 为 8/16/32；`cv2` 输出四条边界各 16 个 logits（`reg_max=16`）。每条边界经 softmax 后以 `sum(k*p_k)` 得到连续距离，再乘 stride 并围绕 offset=0.5 的 anchor point 解码。因此 stride=8 不等于坐标只能按 8 像素量化。

训练时 TaskAlignedAssigner 联合分类分数和预测框 IoU 选择正样本；BboxLoss 对正样本计算 IoU loss 与相邻离散 bin 插值的 DFL loss。极小目标的风险来自低分辨率空间特征、正样本数量/排序以及四边距离估计误差的共同作用，而不是整数网格量化这一单一因素。

本阶段从 Stage10 三折 OOF checkpoint 导出 NMS 前 logits，并分别分析固定推理最终框和 GT 选择的 oracle 候选。后者只用于诊断，绝不作为检测指标。实现见 `stage11_dfl_diagnosis.py`，逐实例结果见 `dfl_instance_diagnosis.csv`。

结论：DFL 本身具备连续定位能力；当前证据不支持“stride8 导致 8 像素量化”的解释。空间信息、候选排序与尺度误差仍需联合考虑。
