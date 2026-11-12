# NBR-DE 实现报告

- 模块：`models/nbr_de.py`
- 网络：`models/yolo11s_nbr_de.yaml`
- 主配置：`configs/e3_yolo11s_nbr_de.yaml`
- 消融：`e3_ablation_no_reference.yaml`、`e3_ablation_no_gate.yaml`
- 位置：YOLO11s Neck 的 P3 融合输出之后
- 新增参数：108,033；模型总参数：9,536,599
- Ultralytics 静态估算：23.0 GFLOPs（未必完整计入张量级稀疏采样运算）
- seed42 峰值训练显存：2.11 GiB

YOLO11s 同语义层通过显式层映射迁移，目标 state elements 迁移比例约 98.86%；未迁移项
为新增 NBR-DE 参数。逐项数值验证通过。模块以运行时注册方式接入 Ultralytics YAML，
没有修改 site-packages。

形状、恒等初始化、边界不回绕、前向、反向、FP32/AMP 和非有限值测试均已通过。
