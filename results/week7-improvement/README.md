# 🧩 协同信息如何改变语义ID？

本周实验已完成：仅用train观看行为构建64维协同向量，与384维文本向量融合，再重新量化为语义ID。四组表示按valid筛选，冻结Fixed后完成三种子正式比较。

| 阶段 | 看什么 |
|---|---|
| [01 · 诊断](01-diagnostics/) | 训练曲线、五种搜索策略与冻结输入 |
| [02 · 构建表示](02-embeddings/) | 输入来源、融合权重与协同近邻 |
| [03 · 生成ID](03-semantic-ids/) | 四组ID、码本使用与重建质量 |
| [04 · 验证筛选](04-screening/) | 四组seed42比较；[冻结选择](selection.json) |
| [05 · 正式训练](05-formal-training/) | Text／Fixed三种子与h20基线的训练证据 |
| [06 · 正式测试](06-evaluation/) | 12项test、分组结果与配对置信区间 |

**Fixed相对受控Text的test NDCG@10提高约17.0%。** 与同h20的GRU4Rec总体接近，差值的95%配对区间包含0；长尾与少样本仍较弱。完整指标以[最终报告](06-evaluation/main_results.md)为准。
