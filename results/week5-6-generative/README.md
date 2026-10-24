# Week 5–6：生成式推荐

本目录按照实验推进顺序保存 T5 式生成推荐模型的训练、评测和汇总证据。

| 目录 | 内容 |
|---|---|
| [`01-pipeline-validation/`](01-pipeline-validation/) | 接口审计、静态检查、冒烟/恢复和小样本过拟合测试 |
| [`02-learning-rate-selection/`](02-learning-rate-selection/) | 两档学习率预实验及选择结论 |
| [`03-formal-training/`](03-formal-training/) | seeds 42/43/44 正式训练日志、验证轨迹与最佳指标 |
| [`04-generative-evaluation/`](04-generative-evaluation/) | valid/test 的约束与无约束生成指标 |
| [`05-baseline-evaluation/`](05-baseline-evaluation/) | SASRec / GRU4Rec 统一口径复评及旧日志对齐 |
| [`06-reports/`](06-reports/) | 最终主表、分组分析与固定用户解码样例 |

主实验采用 3,416 个物品全量排序，并对所有模型统一屏蔽完整历史。生成模型 test NDCG@10 为 `0.1180 ± 0.0092`，低于 GRU4Rec 的 `0.1882 ± 0.0009`。完整结果与边界说明见 [`06-reports/main_results.md`](06-reports/main_results.md)。
