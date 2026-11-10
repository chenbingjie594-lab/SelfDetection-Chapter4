# Plastic_Bomo 第四章数据审计

状态：**BLOCKED**

## 已确认

- 候选冻结 YAML：`DWBG_v2_final_s42/real_only.yaml`
- Train：138 张、168 个实例（Flash 88 / Black 80）
- Val：46 张、64 个实例（Flash 20 / Black 44）
- 全部 184 张图像为 4096×1024
- 类别固定为 `Flash point` 和 `Big black spots`
- 标签均为合法的五字段 YOLO HBB，图像/标签一一对应
- train/val 没有字节级 SHA256 重复
- 文件名未发现生成样本标记；历史 build report 同样声明 Real-only 合成数为 0

## 阻塞问题

按明确文件亲缘规则去除 `_pre_part_0/1` 后，发现两个 canonical/source group 同时
出现在 train 和 val：`img_1020`、`img_9509`。这不是字节重复，但属于已知 sibling
关系；历史泄漏修复文档也报告两个 full-pool source-group overlap。

当前证据不足以证明两个 part 在检测任务中统计独立，因此依据用户指定的保守门禁，
不得正式训练、不得自行重划分、不得以 136/48 或生成实验数据替代。

## 已完成但不构成实验结果的工作

- 全量 SHA256/canonical/source-group manifest
- E0/E1 模型构建与 CPU 前向测试
- E1 兼容权重迁移审计
- E2 坐标正反变换、跨边界覆盖、恢复坐标和全图 NMS 测试
- 仅由训练集冻结的类别尺寸与局部对比度 tertile 阈值

这些结果只证明代码路径的静态正确性，不是检测性能证据。
