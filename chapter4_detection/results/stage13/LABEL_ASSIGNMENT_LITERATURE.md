# 标签分配文献与创新边界

| 方法 | 候选/正样本机制 | 微小目标限制 | 对YOLO11适配 |
|---|---|---|---|
| [ATSS](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Bridging_the_Gap_Between_Anchor-Based_and_Anchor-Free_Detection_via_Adaptive_CVPR_2020_paper.pdf) | 各层中心距离候选，按IoU均值+方差自适应阈值并要求中心在GT内 | 极小框内部点与prior重叠仍稀少 | 需替换TAL，不能视为小改动 |
| [TOOD/TAL](https://openaccess.thecvf.com/content/ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html) | 分类与定位对齐指标选择top-k，并用任务对齐权重监督 | 候选几何池小或早期IoU低时有效权重弱 | YOLO11已采用其核心TAL思想 |
| [YOLOX/SimOTA](https://arxiv.org/abs/2107.08430) | center prior内按分类+回归cost，依据IoU和动态确定正样本数 | 动态k仍不能创造缺失的空间采样点 | 可替换assigner，但训练机制显著改变 |
| [Libra-SOD](https://www.sciencedirect.com/science/article/pii/S0950705124009870) | BLA让不同尺寸GT获得相同数量高质量样本，并配套task-aware head/loss | 固定平衡数量可能引入低质量样本；原论文含多组件 | 不能只复制等量分配并重命名 |

TOOD、ATSS、SimOTA以及高斯中心先验都已是公开成熟思想。Libra-SOD尤其直接研究“小目标正样本不足+平衡分配”，所以CGTA若只加入中心距离或增加小目标正样本，不具备清晰创新差异。

本项目与COCO、航空/遥感小目标不同：原图极宽、输入缩放后多数缺陷一边<16px、仅138张训练图且背景纹理重复。当前数据还显示TAL已有16px候选扩展并选入几乎全部合法候选，因此优先问题不是发明另一种top-k，而是验证原始框外候选的监督有效性与输入几何信息上限。
