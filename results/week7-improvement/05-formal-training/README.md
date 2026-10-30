# 🏋️ 训练记录

已收录四组 seed42 筛选运行、Text / Fixed 各三个种子的正式训练，以及 SASRec / GRU4Rec 各三个种子的历史长度 20 基线。所有正式训练均已完成，test已解锁；正式评测尚未齐备。

## 文件说明

| 文件 | 内容 |
|---|---|
| `{text,fixed,fixed_match,adaptive}_s42_best.json` | 最佳 valid 指标、训练结束标记、配置、数据身份与服务器 checkpoint 路径 |
| `{text,fixed,fixed_match,adaptive}_s42_history.jsonl` | 每 100 步的 loss / 学习率与每 5,000 步的完整 valid 指标 |
| `{text,fixed}_s{43,44}_best.json` / `*_history.jsonl` | Text 与冻结候选的补充种子、最佳指标及完整轨迹 |
| `{text,fixed}-s{43,44}.log` | 补充种子的训练输出；Text seed44 含磁盘故障及从第5,000步恢复的原始记录 |
| `{sasrec,gru4rec}_h20_s{42,43,44}.json` | 基线选模口径、最佳 valid 指标、checkpoint 路径与 SHA-256 |
| `baseline-logs/{sasrec,gru4rec}_h20_s{42,43,44}.log` | 六次基线训练的原始日志，按模型和种子命名 |

`s42` 表示训练种子 42，`h20` 表示模型最多读取最近 20 个物品。基线日志原位于服务器 `log/SASRec/` 或 `log/GRU4Rec/`，本地仅调整文件名与展示目录；内容未改写，日志中的 checkpoint 路径可与同种子 JSON 对照。

## 生成模型三种子验证结果

六个运行均完成60,000步，best也均位于第60,000步。下表是选模用的valid结果，正式test结果另行报告。

| 方案 | seed42 | seed43 | seed44 | 均值 ± 标准差 |
|---|---:|---:|---:|---:|
| Text | 0.17761 | 0.17203 | 0.16203 | 0.17056 ± 0.00644 |
| Fixed | 0.19352 | 0.19724 | 0.19862 | 0.19646 ± 0.00216 |

指标为完整历史屏蔽后的valid NDCG@10，总体标准差ddof=0。Fixed在三个种子上均高于Text，均值相对提高约15.2%；这仍需正式test验证。

## h20 基线验证结果

两种模型均按全目录打分、完整历史屏蔽后的 valid NDCG@10 选模。epoch 为 RecBole 原始编号，从 0 开始。

| 模型 | 种子 | 最佳 epoch | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| SASRec | 42 | 46 | 0.30828 | 0.17786 |
| SASRec | 43 | 27 | 0.31391 | 0.17583 |
| SASRec | 44 | 47 | 0.31474 | 0.17781 |
| GRU4Rec | 42 | 46 | 0.34387 | 0.20071 |
| GRU4Rec | 43 | 38 | 0.33725 | 0.19569 |
| GRU4Rec | 44 | 41 | 0.34272 | 0.19845 |

这些数值用于选模，不能与旧实验的 test 主表直接比较。六个生成模型运行已经齐备，接下来统一进行正式test。

四组生成模型的 best/last 签名、权重有限性与验证轨迹已核对；六个基线的 checkpoint 哈希、配置与日志选模分数一致。大型 checkpoint 和优化器状态保留在服务器，不进入 Git。
