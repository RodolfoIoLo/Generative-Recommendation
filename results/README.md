# 📦 实验产物索引

七周实验已完成。这里按“输入如何变成结果”整理证据；首页只展示最终同口径结论，历史实验保留原有配置与数值。文件名中的`s42`表示随机种子42，`h20`表示最多读取最近20部电影。

JSON保持标准格式，便于程序直接读取；用途和字段解释写在README中。`*_best*.json`记录选模结果与配置，`*_history.jsonl`逐行记录训练轨迹，`*_metrics.json`或正式评测JSON保存排序指标，`*_samples.json`保存固定用户样例，`.log`保留原始运行输出。

<details>
<summary><strong>Week 1 · 数据处理与统计</strong></summary>

| 产物 | 用途 |
|---|---|
| [stats.md](week1-data-stats/stats.md)、`stats.json` | 用户、物品、交互和序列分布 |
| `preprocess_log.json` | 清洗、5-core过滤、划分与数量检查 |
| `item_freq.json` | 全部清洗交互的物品频次，仅作数据描述 |
| [figs/](week1-data-stats/figs/) | 序列长度、物品频次与长尾分布图 |

清洗后6,040名用户、3,416部电影；train有987,531次交互。原始数据、三份划分、ID映射与电影文本在实验数据目录；Week 7分组与融合权重使用单独统计的**train频次**。

</details>

<details>
<summary><strong>Week 2 · SASRec与GRU4Rec基线</strong></summary>

| 产物 | 用途 |
|---|---|
| [baseline_table.md](week2-sasrec-baseline/baseline_table.md) | 两种模型、三个种子的原始基线表 |
| [training-logs/](week2-sasrec-baseline/training-logs/) | 按模型整理的训练输出 |
| [evaluation-logs/](week2-sasrec-baseline/evaluation-logs/) | 已有checkpoint的补充评测输出 |

本周沿用原始RecBole评测设置。完整历史屏蔽后的复评在Week 5–6中，最终h20受控基线在Week 7中；三者分别报告。

</details>

<details>
<summary><strong>Week 3–4 · 文本向量、残差量化与冻结接口</strong></summary>

| 目录 | 产物与用途 |
|---|---|
| [01-quantization/](week3-4-semantic-id/01-quantization/) | 两种方法的`identifiers/`、`logs/`和[坍缩修复记录](week3-4-semantic-id/01-quantization/residual_quantization.md) |
| [02-codebook-analysis/](week3-4-semantic-id/02-codebook-analysis/) | 码本统计、分方法利用率图、[质量报告](week3-4-semantic-id/02-codebook-analysis/codebook_report.md) |
| [03-grid-search/](week3-4-semantic-id/03-grid-search/) | 8组主网格、1组四层对照；候选ID与日志按方法分目录 |
| [04-prefix-validation/](week3-4-semantic-id/04-prefix-validation/) | 共享前缀的语义一致性报告和样例CSV |
| [05-frozen-interface/](week3-4-semantic-id/05-frozen-interface/) | `semantic_ids.json`、三份`*.semantic.jsonl`、`token_vocab.json`、`meta.json`，共六份接口文件 |

正式RQ-VAE重建R²为0.623914，三层使用256/253/253个码字；追加消歧位后，3,416部电影均可唯一查回。`item_emb.npy`和量化checkpoint保留在实验工作区。完整说明见[本周索引](week3-4-semantic-id/README.md)。

</details>

<details>
<summary><strong>Week 5–6 · 生成模型训练与首版评测</strong></summary>

| 目录 | 产物与用途 |
|---|---|
| [01-pipeline-validation/](week5-6-generative/01-pipeline-validation/) | `audit/`审计、`smoke/`小规模运行与恢复、`overfit/`256样本过拟合 |
| [02-learning-rate-selection/](week5-6-generative/02-learning-rate-selection/) | 两档学习率各自的日志、轨迹、最佳valid指标与选择结论 |
| [03-formal-training/](week5-6-generative/03-formal-training/) | 按`seed42/43/44`整理的日志、JSONL轨迹、最佳valid指标 |
| [04-generative-evaluation/](week5-6-generative/04-generative-evaluation/) | 按`valid/test → constrained/unconstrained`整理的12份指标 |
| [05-baseline-evaluation/](week5-6-generative/05-baseline-evaluation/) | 按SASRec/GRU4Rec整理的6份复评指标及旧日志对齐检查 |
| [06-reports/](week5-6-generative/06-reports/) | [主表与分组](week5-6-generative/06-reports/main_results.md)、[20名用户解码样例](week5-6-generative/06-reports/decode_samples.md) |

首版生成模型test NDCG@10为0.1180±0.0092；训练预算为30k。完整预测与checkpoint在实验工作区，仓库保留可阅读的样例。它与Week 7同时改变了输入表示和学习率日程，不作单因素比较。

</details>

<details>
<summary><strong>Week 7 · 协同融合、受控比较与最终结果</strong></summary>

| 目录 | 产物与用途 |
|---|---|
| [01-diagnostics/](week7-improvement/01-diagnostics/) | `environment/`输入与版本指纹，`training-curves/`11次训练的114个验证点，`search/`五种搜索诊断，`cpu-smoke/`运行检查 |
| [02-embeddings/](week7-improvement/02-embeddings/) | `meta.json`记录train-only来源和向量形状，`weights.json`记录四组权重，`neighbors.json`展示协同近邻 |
| [03-semantic-ids/](week7-improvement/03-semantic-ids/) | 四组各自的`identifiers/`映射、`quality/`量化与前缀统计、`training/`日志、`figures/`分配图 |
| [04-screening/](week7-improvement/04-screening/) | 四组seed42原始训练输出、`ablation.json`验证集消融 |
| [selection.json](week7-improvement/selection.json) | 仅按valid冻结的Fixed候选、控制变量与数据身份 |
| [05-formal-training/](week7-improvement/05-formal-training/) | `generative/方案/seed/`中的8次训练记录；`baselines/模型/seed/`中的6次基线训练记录 |
| [06-evaluation/](week7-improvement/06-evaluation/) | 生成式／基线分别按模型和种子存放12项test指标与20用户样例；`summaries/`含分组表及两组配对区间，`figures/`含最终对比图 |

四种表示均完成量化；Fixed在valid筛选中胜出，随后与Text各补齐三个种子。正式比较覆盖四种模型×三个种子，每项6,040名用户；全部Top-10合法、互异且未出现在完整历史中。[最终报告](week7-improvement/06-evaluation/main_results.md)包含结论与证据边界。

</details>

## 文件在哪里？

本仓库存放代码、配置、统计、日志、指标和轻量图表；实验工作区另有原始数据、`.npy`向量、完整预测与大型模型权重。对应关系见[服务器目录](../docs/server-layout.md)，重新生成这些产物见[复现指南](../docs/reproduction.md)。原始日志和冻结元数据保留运行时的命名，脚本通过统一路径规则读取整理后的结果，不改写历史记录。
