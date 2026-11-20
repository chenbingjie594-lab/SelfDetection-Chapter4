# Stage20 最终总结

## 核心结论

1. 82 个高质量候选生成失败 GT 中，80 个有缓存候选。失败主要表现为实例级尺寸/边界误差，而非纯中心偏移。
2. 中心修正恢复 33.75%，尺寸修正恢复 60.0%；宽高误差与 IoU 的相关强度也明显高于中心误差。
3. 上下边成对修正恢复 87.5%，高于左右边的 73.75%；总体只有轻微横向收缩，不存在足以用常数解决的统一偏差。
4. Flash、Black 和极小子集均呈现“上下边界 > 尺寸 > 中心”的反事实趋势。极小目标误差更大，但没有证据支持完全不同的类别机制。
5. P3 缺陷边界响应明显强于普通背景，不过中心加四边采样的分组探针没有带来增量几何预测能力（MAE 0.0680→0.0740）。
6. 因机制门槛失败，P3-BAR 未实现、未训练；问题 8～11（训练保存、高质量候选覆盖、AP75/mAP、关闭分支干预）均为“不适用/未运行”，没有虚构结果。
7. P3-BAR 目前不值得作为第四章正式创新候选。后续若继续定位方向，应先寻找能在训练集分组验证中稳定预测有符号边界残差的表征或监督证据，而不是围绕采样点和残差结构反复试验。

## 产物

- `geometry_by_gt_selection.csv`：oracle、最高分和 NMS 保留候选逐 GT 几何误差。
- `counterfactual_box_repairs.csv`：全部预列明反事实修正。
- `geometry_summary.json`、`geometry_error_distributions.png`。
- `p3_boundary_instances.csv`、`p3_boundary_summary.json`、`p3_boundary_information.png`。
- `environment_audit.json`、`artifact_hashes.sha256`、`RUN_COMMANDS.md`。

当前阶段只完成 CPU 诊断和单元测试；CUDA 驱动不可见，且机制门槛本身未通过，因此没有正式训练任务需要续跑。
