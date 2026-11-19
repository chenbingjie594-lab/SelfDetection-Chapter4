# 候选筛选追踪

## 协议与一致性

使用 Stage10 冻结三折、seed42、last.pt 对应的 Stage11 原始缓存，严格复现 Ultralytics 8.4.145 `DetectionValidator`：`conf=0.001`、multi-label、class-aware NMS、IoU=0.7、`max_nms=30000`、`max_det=300`。TraceableNMS 保留 anchor/class 索引及抑制者。

在 E0/CBCD 合计 276 个模型×OOF 图像上，官方与追踪输出的数量、顺序、类别、分数、坐标和 anchor index 全部一致，最大绝对差为 0。由追踪输出重新评价得到：E0 mAP50=.806085、mAP50-95=.353951、AP75=.244184；CBCD 为 .832899、.360453、.240388，精确复现 Stage10 OOF。

## 筛选计数

| 模型 | 原始 anchors | 低于预过滤 | 通过预过滤 anchors | class候选对 | 最终保留 | NMS抑制 | max_nms/max_det |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | 1,846,026 | 1,844,929 | 1,097 | 1,274 | 744 | 530 | 0 |
| CBCD | 1,846,026 | 1,844,981 | 1,045 | 1,231 | 668 | 563 | 0 |

被抑制框与抑制框重叠度中位数约为 E0 .846、CBCD .846，均严格经过官方 class-aware NMS 路径，而非按位置猜测。完整低分候选保存在 880MB 本地 `.npz`；2,505 个阈值以上 candidate/class 对及原因见 `candidate_trace.csv`/`.jsonl.gz`。各缓存 SHA256 在 `analysis_summary.json`。
