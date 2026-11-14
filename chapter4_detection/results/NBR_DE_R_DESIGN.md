# NBR-DE-R 设计与冻结协议

E4 在 E3 相同 P3 位置插入一个可靠性引导模块，不改变检测头、损失或训练策略。它同时计算中心排除的稀疏 Attention 参考 `B_att` 与 3×3 深度卷积参考 `B_local`。

归一化邻域不一致度：

`U = Σ α_j ||V_j-B_att||²/C / (Σ α_j ||V_j||²/C + ε)`

可靠性权重：

`r = sigmoid(b - softplus(k_raw) U)`

融合参考及输出：

`B = r B_att + (1-r) B_local`

`F' = F + gamma G D`

`b=0`、`k_raw=0`，`gamma=0`，因此初始网络严格保持恒等映射。U 增大时 r 不增加，边界无效邻点在 softmax 前屏蔽。正式初始化只从原始 YOLO11s 做语义迁移，没有加载 E3 权重。

模型总参数 9,536,601，NBR-DE-R 参数 108,035。测试覆盖前后向、FP32/AMP、尺寸、可靠性边界和单调性、融合公式、边界采样、恒等初始化及 checkpoint 重载。

正式复现命令：

```bash
/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python -m chapter4_detection.scripts.train \
  --experiment E4 --seed 42 --config chapter4_detection/configs/e4_nbr_de_r.yaml --device 0
```
