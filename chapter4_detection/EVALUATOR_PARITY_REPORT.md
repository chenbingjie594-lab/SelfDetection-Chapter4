# E0/E1/E2 评价器一致性报告

## 结论

统一评估已通过。E0 与 E1 的验证器内存预测经 `evaluate_original.py` 重算后，
Precision、Recall、mAP50、mAP50-95、AP75 与 Ultralytics 8.4.145 原生结果的绝对差均为 0。

## 已定位的问题

早期逐图 `model.predict()` 与 `model.val()` 的 letterbox/batch 路径不同，E1 mAP50
曾偏差 0.0345。改用验证器内存预测后，还发现常规 `scale_boxes()` 会裁剪原图边界外
的框，而原生验证是在裁剪前匹配；该操作会改变边缘微小目标 IoU。最终导出采用同一
ratio/pad 的逆变换但不裁剪，保持 IoU 几何不变。

测试覆盖空预测、空 GT、错误类别和重复预测；相关测试全部通过。E2 原图级指标因而
可以与 E0/E1 使用同一 AP 实现比较，但 E2 的输入协议仍属于不同对照。
