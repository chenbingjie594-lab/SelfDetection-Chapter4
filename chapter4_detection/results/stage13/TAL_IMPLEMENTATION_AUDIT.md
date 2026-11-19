# TAL 实现审计

审计对象是本地 Ultralytics 8.4.145，而非最新版源码。运行中 `v8DetectionLoss` 实际创建 `TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0, stride=[8,16,32])`，`reg_max=16`。

流程如下：

1. `make_anchors(..., offset=0.5)` 生成 P3/P4/P5 网格中心。
2. `select_candidates_in_gts` 先把任一小于 `stride_val=stride[1]=16` 的 GT 边长扩为16，中心不变，再用严格内部判定（四边距离均大于 `1e-9`）。这是当前安装版本的重要实现细节。
3. 仅对内部候选计算 CIoU（截断到非负）和 `score^0.5 * CIoU^6`。
4. 每个 GT 取 top-10，但之后再与内部 mask 相乘；因此 top-k不是保证10个正样本。合法候选少于10时最多只能取合法数量；预测重叠为零时仍可能形成 foreground，但归一化 target score 可为零，不能产生有效 box/DFL 权重。
5. 多 GT 竞争同一 anchor 时，按最高 overlap 指派，而非按分类分数；随后可选 `topk2`，本任务中等于 topk，不触发二次筛选。
6. 每个 GT 的 alignment metric 以该 GT 的最大 overlap/最大 alignment 归一化，形成 soft target score。分类 BCE、CIoU box loss、DFL loss均按 target score 总和归一化；box/DFL 单正样本权重为该 anchor 的 target-score和。

无内部 anchor 时，该 GT 的正样本与有效回归监督均为0。低 IoU使六次幂 alignment 很小；即使 foreground mask为真，target score接近0时监督仍可数值退化。

监控器不修改 site-packages。合成一致性测试中 target labels/boxes/scores、fg mask、GT index、构造损失和梯度逐元素一致；完整150轮诊断训练的逐 epoch loss/指标与 Stage10 Fold0 基线完全一致，最终499个 state tensor逐元素相同（最大差0），进一步验证监控未改变训练行为。
