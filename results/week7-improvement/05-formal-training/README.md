# 🏋️ 训练记录

已收录四组 seed42 筛选运行，以及 SASRec / GRU4Rec 各三个种子的历史长度 20 基线。Text 与 Fixed 的 seed43/44 尚未全部完成，进行中的记录暂不作为完整成果入库；第七周正式 test 尚未开始。

## 文件说明

| 文件 | 内容 |
|---|---|
| `{text,fixed,fixed_match,adaptive}_s42_best.json` | 最佳 valid 指标、训练结束标记、配置、数据身份与服务器 checkpoint 路径 |
| `{text,fixed,fixed_match,adaptive}_s42_history.jsonl` | 每 100 步的 loss / 学习率与每 5,000 步的完整 valid 指标 |
| `{sasrec,gru4rec}_h20_s{42,43,44}.json` | 基线选模口径、最佳 valid 指标、checkpoint 路径与 SHA-256 |
| `baseline-logs/{sasrec,gru4rec}_h20_s{42,43,44}.log` | 六次基线训练的原始日志，按模型和种子命名 |

`s42` 表示训练种子 42，`h20` 表示模型最多读取最近 20 个物品。基线日志原位于服务器 `log/SASRec/` 或 `log/GRU4Rec/`，本地仅调整文件名与展示目录；内容未改写，日志中的 checkpoint 路径可与同种子 JSON 对照。

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

这些数值用于选模，不能与旧实验的 test 主表直接比较。正式 test 将在 Text 与 Fixed 的六个训练运行完成后统一执行。

四组生成模型的 best/last 签名、权重有限性与验证轨迹已核对；六个基线的 checkpoint 哈希、配置与日志选模分数一致。大型 checkpoint 和优化器状态保留在服务器，不进入 Git。
