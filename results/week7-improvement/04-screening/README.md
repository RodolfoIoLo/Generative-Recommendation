# 🎯 只用验证集选择方案

四组模型采用相同架构、seed42、60k预算和解码设置，只改变输入表示。

| 表示 | valid NDCG@10 |
|---|---:|
| Text | 0.177611 |
| Fixed | **0.193518** |
| Fixed-Match | 0.178134 |
| Adaptive | 0.190177 |

按valid冻结Fixed，随后只为Text和Fixed补齐seed43/44并进入正式test。Adaptive未在这次筛选中超过Fixed，单seed不支持稳定的策略优劣结论。

[ablation.json](ablation.json)保留全部分组指标，`training-logs/`保留四组原始输出；[selection.json](../selection.json)记录选择、控制变量与输入身份。[正式测试](../06-evaluation/main_results.md)单独报告。
