# 🧪 四组 seed42 验证集筛选

四组均训练至 60,000 步，模型结构、训练日程与解码设置一致；使用全部 6,040 名 valid 用户，按完整历史屏蔽后的 NDCG@10 选择 checkpoint。

| 表示 | 最佳步数 | Recall@10 | NDCG@10 | 较 Text 的 NDCG@10 差值 |
|---|---:|---:|---:|---:|
| Text | 60,000 | 0.30844 | 0.17761 | — |
| **Fixed** | **60,000** | **0.34007** | **0.19352** | **+0.01591** |
| Fixed-Match | 55,000 | 0.31805 | 0.17813 | +0.00052 |
| Adaptive | 60,000 | 0.33195 | 0.19018 | +0.01257 |

Fixed 的 NDCG@10 较 Text 相对提高约 8.96%，达到预设的绝对差值 0.003 筛选门槛，因此冻结为后续候选。完整记录见 [selection.json](../selection.json)，各组分组指标见 [ablation.json](ablation.json)。

这只是单种子验证集筛选，不是最终 test 或显著性结论。频次自适应在本轮没有超过固定权重；额外收益仍需后续三种子和 test 检验。

四份 `*-s42.log` 为原始训练输出；对应的 `*_s42_best.json` 和 `*_s42_history.jsonl` 位于 [05-formal-training](../05-formal-training/README.md)，保留训练与选择脚本使用的相对路径。[验证曲线](../01-diagnostics/training_curves.png)可用于查看收敛趋势。
