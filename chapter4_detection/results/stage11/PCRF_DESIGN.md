# PCRF 设计冻结

PCRF（Phase-Consistent Regression Fusion）是单一探索候选，不预先主张原创性。它计算 `R2=P2-DWAvg3x3(P2)`，用 PixelUnshuffle(2) 对齐至 P3，经 1x1 投影产生小幅 `delta`，只令 P3 回归输入为 `P3+delta`。P3 分类输入、P4/P5、reg_max、DFL 解码和损失均不变。

固定平均核为 buffer；投影权重以标准差 1e-4 的非零正态初始化。奇数尺寸先裁到目标两倍大小，不足则右/下 padding。开发门槛依据 P2 更近的空间峰值、弱但存在的几何相关和当前宽高误差；P2 较低的缺陷特异性被明确视作风险。
