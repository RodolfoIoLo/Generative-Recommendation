# 实验结果与数据产物

## 总览

| 周次 | 结果目录 / 主要产物 | 状态 |
|---|---|---|
| Week 1 | `week1-data-stats/`；预处理数据位于服务器 `/data/zby/data/ml-1m/processed/` | ✅ 完成 |
| Week 2 | `week2-sasrec-baseline/`；checkpoint 位于服务器 `~/genrec/saved/` | ✅ 完成 |
| Week 3–4 | 文本编码、残差量化、码本统计、前缀验证与接口冻结 | ✅ 实验与文档已完成，待 Git 提交 |
| Week 5–6 | 生成式模型训练日志与对比表 | ⏳ 未开始 |
| Week 7 | 冷启动 / 长尾实验、消融与最终报告 | ⏳ 未开始 |

约定：

- **入库**：体积小、适合审阅的统计、干净日志、表格和图片。
- **服务器保留**：原始数据、可重建的中间数组、checkpoint、TensorBoard 事件和带进度条的原始日志。
- RecBole 日志中的用户 / 物品数量会包含 1 个 PAD token；真实规模仍是 **6,040 用户、3,416 物品**。

## Week 1：数据准备与基础分析

### 数据结论

| 指标 | 结果 |
|---|---:|
| 原始评分 | 1,000,209 |
| 5-core 后交互 | 999,611 |
| 用户 / 物品 | 6,040 / 3,416 |
| train / valid / test 交互 | 987,531 / 6,040 / 6,040 |
| 平均 / 中位序列长度 | 165.5 / 96 |
| 序列长度 P90 / 最大值 | 400 / 2,277 |
| 物品流行度 Gini | 0.6034 |
| Top 10% 物品覆盖的交互 | 42.31% |
| 时间跨度 | 2000-04-25 ～ 2003-02-28 |

### 仓库内文件

| 文件 | 生成者 | 内容 |
|---|---|---|
| [`preprocess_log.json`](week1-data-stats/preprocess_log.json) | `scripts/preprocess.py` | raw → 5-core → split 的规模记录 |
| [`stats.md`](week1-data-stats/stats.md) | `scripts/stats.py` | 人类可读的数据统计与抽样序列 |
| [`stats.json`](week1-data-stats/stats.json) | `scripts/stats.py` | 机器可读统计 |
| [`item_freq.json`](week1-data-stats/item_freq.json) | `scripts/export_item_freq.py` | 3,416 个物品的交互频次，供长尾评测使用 |
| [`item_freq_hist.png`](week1-data-stats/figs/item_freq_hist.png) | `scripts/stats.py` | 物品热度直方图 |
| [`item_freq_loglog.png`](week1-data-stats/figs/item_freq_loglog.png) | `scripts/stats.py` | 物品热度双对数图 |
| [`seq_len_hist.png`](week1-data-stats/figs/seq_len_hist.png) | `scripts/stats.py` | 用户序列长度分布 |

### 服务器数据文件

原始数据位于 `/data/zby/data/ml-1m/raw/`：

| 文件 | 规模 | 用途 |
|---|---:|---|
| `ml-1m.zip` | 5.9 MB | GroupLens 原始压缩包 |
| `README` | 5.6 KB | 数据集官方字段说明 |
| `ratings.dat` | 24.6 MB；1,000,209 行 | `UserID::MovieID::Rating::Timestamp`，唯一行为数据源 |
| `movies.dat` | 171 KB；3,883 行 | `MovieID::Title::Genres`，唯一物品文本来源 |
| `users.dat` | 134 KB；6,040 行 | 用户画像；当前协议不使用 |

`scripts/preprocess.py` 生成 `/data/zby/data/ml-1m/processed/`：

| 文件 | 规模 / 形状 | 下游用途 |
|---|---|---|
| `ml-1m.inter` | 21.5 MB；999,611 条交互 + 表头 | RecBole 的 SASRec / GRU4Rec 输入 |
| `train.jsonl` | 5.7 MB；6,040 行 | 每用户一条训练序列，共 987,531 次训练交互 |
| `valid.jsonl` | 5.8 MB；6,040 行 | `history + target`，用于模型选择 |
| `test.jsonl` | 5.9 MB；6,040 行 | `history + target`，用于最终评测 |
| `user_id_map.json` | 94 KB；6,040 个映射 | 原始用户 ID → 连续 ID `0..6039` |
| `item_id_map.json` | 53 KB；3,416 个映射 | 原始物品 ID → 连续 ID `0..3415` |
| `item_text.json` | 179 KB；3,416 条文本 | Week 3 文本编码输入 |
| `preprocess_log.json` | 299 B | 数据版本与规模校验 |
| `stats/stats.md` | 4.4 KB | 服务器端统计报告 |
| `stats/stats.json` | 568 B | 服务器端机器可读统计 |
| `stats/figs/*.png` | 3 张图 | 与仓库 `week1-data-stats/figs/` 对应 |

RecBole 通过软链接读取同一份数据：

```text
/data/zby/data/recbole/ml-1m -> /data/zby/data/ml-1m/processed
```

### 最小真实样例

```text
# ratings.dat
1::1193::5::978300760

# ml-1m.inter（TAB 分隔；ID 已重映射）
userID:token    itemID:token    timestamp:float
0               2757            978300019.0
```

```json
// train.jsonl
{"user": 0, "items": [2757, 861, 1068, "..."]}

// valid/test.jsonl
{"user": 0, "history": [2757, 861, 1068, "..."], "target": 1578}

// item_text.json
{"0": "Toy Story (1995) | Animation|Children's|Comedy"}
```

`train.items = valid.history`，而 `test.history = valid.history + valid.target`；`scripts/verify_split.py`
已逐用户检查时间单调、目标位置与历史前缀，结果为 **PASS**。

## Week 2：SASRec / GRU4Rec 基线

### 测试集结果

评测为 3,416 物品全量排序。SASRec 的标准差使用 `numpy.std(ddof=0)`。

| 模型 | Seed | 总 epoch | Best epoch | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SASRec | 42 | 43 | 31 | 0.1409 | 0.2204 | 0.0884 | 0.1141 |
| SASRec | 43 | 109 | 97 | 0.1523 | 0.2339 | 0.0968 | 0.1230 |
| SASRec | 44 | 59 | 47 | 0.1526 | 0.2250 | 0.0958 | 0.1191 |
| **SASRec** | **均值 ± std** | — | — | **0.1486 ± 0.0054** | **0.2264 ± 0.0056** | **0.0937 ± 0.0037** | **0.1187 ± 0.0036** |
| **GRU4Rec** | **42** | **67** | **55** | **0.1861** | **0.2657** | **0.1261** | **0.1517** |

独立汇总表见 [`baseline_table.md`](week2-sasrec-baseline/baseline_table.md)。

### 仓库内文件

| 文件 | 来源 | 记录内容 |
|---|---|---|
| [`baseline_table.md`](week2-sasrec-baseline/baseline_table.md) | 从下列日志逐项汇总 | 测试指标、epoch 与协议 |
| `sasrec_s42.log` | `log/SASRec/SASRec-ml-1m-Sep-28-2026_19-17-25-dbfc42.log` | seed 42 完整训练日志 |
| `sasrec_s42_patch_test.log` | `log/SASRec/SASRec-ml-1m-Oct-10-2026_15-55-25-dbfc42.log` | 从 seed 42 best checkpoint 补跑 valid / test |
| `sasrec_s43.log` | `log/SASRec/SASRec-ml-1m-Oct-15-2026_13-00-11-6e006f.log` | seed 43 完整训练与测试日志 |
| `sasrec_s44.log` | `log/SASRec/SASRec-ml-1m-Oct-15-2026_17-48-15-37672d.log` | seed 44 完整训练与测试日志 |
| `gru4rec_s42.log` | `log/GRU4Rec/GRU4Rec-ml-1m-Oct-15-2026_09-39-08-c5973e.log` | GRU4Rec seed 42 完整训练与测试日志 |

seed 42 的 `eval_only.py` 结果与 best epoch 的 valid 指标逐位一致，验证了 checkpoint 读取与评测链路。

### 服务器保留、不进入 Git 的文件

| 位置 | 当前文件 | 原因 |
|---|---|---|
| `~/genrec/saved/` | 3 个 SASRec `.pth` + 1 个 GRU4Rec `.pth` | 二进制 checkpoint；用于复评与断点恢复 |
| `~/genrec/results/` | `sasrec_s42.log`、`sasrec_s42_patch_test.log`、`sasrec_s43.log`、`sasrec_s44.log`、`gru4rec_s42.log` | `tee` 捕获的原始 stdout / stderr，含 tqdm 控制字符 |
| `~/genrec/log_tensorboard/` | 4 个训练 run + 1 个 eval-only run(seed = 42时补跑的test) | TensorBoard 二进制事件 |

仓库只保留 RecBole 自写的干净文本日志；服务器路径和确切文件名见根目录 [`README.md`](../README.md) 的服务器目录树。

## Week 3–4：物品语义标识符

### 文本编码

| 项 | 结果 |
|---|---|
| 输入 | `item_text.json`，3,416 条 `title (year) \| genres` |
| 编码器 | `/data/LLMs/all-MiniLM-L6-v2` |
| 输出 | 服务器 `~/genrec/item_emb.npy` |
| 形状 / dtype | `(3416, 384)` / `float32` |
| L2 范数 min / mean / max | 0.99999982 / 1.00000000 / 1.00000012 |
| NaN / Inf / 全零行 | 0 / 0 / 0 |
| SHA-256 | `162d6d2a9fb89e6b9e096d97a79a46338d83507ad5cbc07581a730d43e2b77a4` |
| 语义 sanity check | 同组余弦 0.5491；随机配对 0.2774；差值 +0.2717 |
| 案例 | Toy Story (1995) / Toy Story 2 (1999) 余弦 0.8027 |

`item_emb.npy` 是可由 `scripts/encode_items.py` 重建的中间数组，因此不进入 Git。

### 残差量化与码本质量

主设置均为 `L=3, K=256`；末位是按冲突组追加的确定性序号，不属于量化码本。

| 方法 | 重建 R² | 三层利用数 | 三层困惑度 | 消歧前冲突率 | 消歧后唯一 ID |
|---|---:|---|---|---:|---:|
| RQ-KMeans | — | 256 / 256 / 256 | 201.0 / 97.4 / 14.9 | 2.78% | 3,416 |
| RQ-VAE | 0.623914 | 256 / 253 / 253 | 219.3 / 232.0 / 241.1 | 0.15% | 3,416 |

RQ-VAE 使用“AE 预训练 → 潜空间残差 K-Means 初始化 → 联合训练”，修复了随机初始化造成的码本坍塌。完整过程和独立复验见 [`residual_quantization.md`](week3-4-semantic-id/residual_quantization.md)。

表中的 `—` 表示原始 RQ-KMeans 脚本未记录 R²，并非该指标不可计算。以三层聚类中心之和作为重建向量，对正式码本只读复算得到 `R²=0.595971`、mean-MSE `0.00075384`；该补充值未参与原实验的模型选择。

### `(L,K)` 网格与冲突消解

8 组主网格全部完成。主网格中 `RQ-VAE, L=3, K=256` 最优：R² 为 0.6245，消歧前冲突率 0.18%；追加序号后全部唯一。额外的 4 层纯 RQ-VAE 不使用序号消歧，也得到 3,416 个唯一 ID，R² 为 0.6566。详见 [`grid_search.md`](week3-4-semantic-id/grid_search.md)。

### 前缀语义一致性与冻结接口

正式方案采用 `L=3, K=256` 的 RQ-VAE 前缀，并追加一位确定性消歧码。共享前缀越长，组内文本向量越相似：

| 前缀深度 | 同前缀平均余弦 | 随机平均余弦 | 差值（95% pair-bootstrap CI） |
|---|---:|---:|---:|
| 1 | 0.5289 | 0.2832 | +0.2456 `[0.2436, 0.2476]` |
| 2 | 0.7116 | 0.2832 | +0.4284 `[0.4079, 0.4498]` |
| 3 | 0.8089 | 0.2832 | +0.5257 `[0.4101, 0.6314]` |

完整统计、10 组人工正例和 2 组反例见 [`prefix_consistency.md`](week3-4-semantic-id/prefix_consistency.md)，原始抽样见 [`prefix_samples.csv`](week3-4-semantic-id/prefix_samples.csv)。四层纯语义候选得到近似的前缀指标，但没有正式可复验 checkpoint，因此保留为后续消融方案。阶段结论统一汇总于 [`codebook_report.md`](week3-4-semantic-id/codebook_report.md)。

`frozen/` 是下一阶段生成模型的固定输入：

| 文件 | 内容 |
|---|---|
| `semantic_ids.json` | 3,416 个物品到四元组语义 ID 的映射 |
| `train/valid/test.semantic.jsonl` | 三个 split 的整数物品 ID 已替换为语义四元组；各 6,040 行 |
| `token_vocab.json` | 1,024 个分层物品 token + 3 个特殊 token |
| `meta.json` | 协议、形状、来源与 embedding/checkpoint/split 的 SHA-256 |

导出脚本的内置检查和独立逐行复验均通过：四元组全部唯一、码值位于 `0..255`、三个 split 与原文件逐项对应，且 `meta.json` 中全部哈希可重算一致。

### 仓库内产物

| 路径 | 内容 |
|---|---|
| `semantic_ids_kmeans.json` / `semantic_ids_rqvae.json` | 阶段 3 正式语义 ID |
| `rq_kmeans.log` / `rqvae_train.log` | 正式运行日志 |
| `codebook_stats.json` / `figs/*.png` | 码本统计与 6 张频次图 |
| `grid_search.md` | 8 组网格、4 层备选与结论 |
| `grid/semantic_ids_*.json` | 9 组对照实验的语义 ID |
| `grid/logs/*.log` / `grid_search.log` | 独立日志与合并日志 |
| `prefix_consistency.md` / `prefix_samples.csv` | 前缀语义一致性统计与人工检查样本 |
| `codebook_report.md` | 码本质量、参数网格、前缀验证、冻结决策与局限性总报告 |
| `frozen/` | 面向生成模型的 6 类冻结接口文件 |

二进制量化器、checkpoint 和可重建的 `item_emb.npy` 留在服务器 `~/genrec/`，不进入 Git。

## Week 5–7

尚无实验文件。后续结果将继续按周放入：

```text
results/
├── week5-6-generative-model/
└── week7-ablation-and-final/
```

## 复现入口

| 目标 | 命令 |
|---|---|
| 生成 Week 1 数据 | `python scripts/preprocess.py --dataset ml-1m` |
| 生成统计与图片 | `python scripts/stats.py --dataset ml-1m` |
| 校验数据划分 | `python scripts/verify_split.py --data-dir /data/zby/data/ml-1m/processed` |
| 导出物品频次 | `python scripts/export_item_freq.py --out item_freq.json` |
| 训练基线 | `python scripts/run_baseline.py --model SASRec --config config/sasrec_ml1m.yaml --seed 42` |
| checkpoint 复评 | `python scripts/eval_only.py --model_file saved/<checkpoint>.pth` |
| 编码物品文本 | `python scripts/encode_items.py --model /data/LLMs/all-MiniLM-L6-v2 --out item_emb.npy` |
| 构建语义 ID | `python scripts/build_semantic_id.py --config config/rqvae_ml1m.yaml --emb item_emb.npy --method rqvae --device cuda:0 --out semantic_ids_rqvae.json` |
| 分析码本 | `python scripts/analyze_codebook.py --kmeans semantic_ids_kmeans.json --rqvae semantic_ids_rqvae.json --outdir results/week3-4-semantic-id` |
| 验证前缀语义 | `python scripts/check_prefix_consistency.py --emb item_emb.npy --ids semantic_ids_rqvae.json --text /data/zby/data/ml-1m/processed/item_text.json --outdir results/week3-4-semantic-id` |
| 冻结生成模型接口 | `python scripts/export_semantic.py --ids semantic_ids_rqvae.json --emb item_emb.npy --quantizer_checkpoint saved/rqvae_ml1m.pt --processed /data/zby/data/ml-1m/processed --outdir exported` |
