# Week 3–4：语义标识符

本目录记录从 384 维物品文本向量到四位语义 ID 的完整过程。正式方案为 `L=3, K=256` 的 RQ-VAE，并追加一位确定性冲突消解码。

| 目录 | 内容 |
|---|---|
| [`01-quantization/`](01-quantization/) | RQ-KMeans / RQ-VAE 正式运行、语义 ID 与残差量化说明 |
| [`02-codebook-analysis/`](02-codebook-analysis/) | 码本统计、利用率图和阶段总报告 |
| [`03-grid-search/`](03-grid-search/) | `(L,K)` 网格实验、候选语义 ID 与逐组日志 |
| [`04-prefix-validation/`](04-prefix-validation/) | 前缀语义一致性统计与人工样例 |
| [`05-frozen-interface/`](05-frozen-interface/) | 供生成模型读取的冻结数据接口与哈希元数据 |

核心结论：正式 RQ-VAE 的重建 R² 为 `0.623914`，三层码本分别使用 `256/253/253` 个码字；消歧后 3,416 个物品均具有唯一语义 ID。详细结论见 [`02-codebook-analysis/codebook_report.md`](02-codebook-analysis/codebook_report.md)，下游推荐结果见 [`Week 5–6 主实验`](../week5-6-generative/06-reports/main_results.md)。
