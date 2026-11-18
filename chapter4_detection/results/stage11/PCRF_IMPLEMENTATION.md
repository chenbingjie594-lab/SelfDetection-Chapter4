# PCRF 实现与初始化审计

实现位于 `models/pcrf.py`，配置为 `configs/e7_pcrf.yaml`。PCRFDetect 复用原 Detect 的 cv2/cv3/dfl，并将 Backbone layer 2 加入保存输出；自定义 trainer 保证训练加载的是替换后的结构。

初始化审计：499 个共享 tensor 与 E0 数值完全一致；分类 raw output 最大绝对差为 0；回归 raw output 相对 L2 变化 0.001374。保存重载后仍为自定义 head。初始化 checkpoint SHA256 为 `7f108f1a74df8e695c57da912342351410239d0b0fc25d4a186c3ff1c6a4e0a1`。

模型为 9,479,110 参数。Ultralytics 标准 640×640 summary 为 22.3 GFLOPs；按空间面积折算，1536×1536 约 128.4 GFLOPs（估算值，未把数据管线计入）。单元测试覆盖输出形状、分类不变、小幅回归变化、几何分支有限非零梯度及 checkpoint 重载。

第一次启动因相对输出路径与 buffer 冻结处理问题在首个 batch 前失败，无训练指标；证据保存在 `invalid_runs/path_bug/`。修复为绝对路径和 registered buffer 后才启动三折正式运行。
