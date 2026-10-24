# 序列推荐基线

> MovieLens 1M，3,416 个物品全量排序；Recall@5/10 与 NDCG@5/10。下表保留 RecBole 原始序列评测口径；统一屏蔽完整历史的公平复评见 [`../week5-6-generative/06-reports/main_results.md`](../week5-6-generative/06-reports/main_results.md)。
> SASRec 与 GRU4Rec 汇总均使用总体标准差 `numpy.std(ddof=0)`。

| 模型 | Seed | 总 epoch | Best epoch | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SASRec | 42 | 43 | 31 | 0.1409 | 0.2204 | 0.0884 | 0.1141 |
| SASRec | 43 | 109 | 97 | 0.1523 | 0.2339 | 0.0968 | 0.1230 |
| SASRec | 44 | 59 | 47 | 0.1526 | 0.2250 | 0.0958 | 0.1191 |
| **SASRec** | **均值 ± std** | — | — | **0.1486 ± 0.0054** | **0.2264 ± 0.0056** | **0.0937 ± 0.0037** | **0.1187 ± 0.0036** |
| GRU4Rec | 42 | 67 | 55 | 0.1861 | 0.2657 | 0.1261 | 0.1517 |
| GRU4Rec | 43 | 71 | 59 | 0.1836 | 0.2624 | 0.1258 | 0.1512 |
| GRU4Rec | 44 | 80 | 68 | 0.1796 | 0.2613 | 0.1235 | 0.1498 |
| **GRU4Rec** | **均值 ± std** | — | — | **0.1831 ± 0.0027** | **0.2631 ± 0.0019** | **0.1251 ± 0.0012** | **0.1509 ± 0.0008** |

## 固定协议

- RecBole 1.2.1，隐式反馈，CE loss。
- 每个用户按时间 leave-one-out：最后一条 test，倒数第二条 valid。
- 最大序列长度 200；valid NDCG@10 早停，`stopping_step=10`。
- RecBole 1.2.1 的原始序列全量排序未额外屏蔽历史物品；Week 5–6 已复现这些数值，并另行计算屏蔽完整历史的主实验结果。

## 证据文件

| 文件 | 内容 |
|---|---|
| `sasrec_s42.log` | seed 42 训练；best epoch 31 |
| `sasrec_s42_patch_test.log` | seed 42 checkpoint 的独立 valid / test 复评 |
| `sasrec_s43.log` | seed 43 训练与测试；best epoch 97 |
| `sasrec_s44.log` | seed 44 训练与测试；best epoch 47 |
| `gru4rec_s42.log` | GRU4Rec seed 42 训练与测试；best epoch 55 |
| `gru4rec_s43.log` | GRU4Rec seed 43 训练与测试；best epoch 59 |
| `gru4rec_s44.log` | GRU4Rec seed 44 训练与测试；best epoch 68 |
