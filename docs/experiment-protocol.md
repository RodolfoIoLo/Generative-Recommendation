# 🔬 实验协议

## 数据与评测

| 项目 | 约定 |
|---|---|
| 数据 | MovieLens 1M，隐式交互，迭代 5-core |
| 划分 | 按时间排序，每用户最后一次为 test、倒数第二次为 valid |
| 规模 | 6,040 用户、3,416 物品，train 有 987,531 次交互 |
| 生成训练样本 | 对 train 序列构造 981,491 个前缀—下一物品样本 |
| 评测 | 全目录候选、无负采样，屏蔽用户完整历史；Recall/NDCG@5/10 |
| 分组 | 根据目标电影的 train 频次划分 few-shot、tail、head |

严格冷启动指 train 中没有交互的物品；当前数据集此类测试目标为 0，记为 N/A。few-shot（train 频次 ≤20）是少样本代理，不等同于严格冷启动。

判别式模型对全部候选打分；生成式模型以动态Trie和beam=20近似检索。全目录指候选空间包含所有物品，不代表生成模型逐一精确打分；搜索诊断另与exact目录打分比较。

## 已完成的生成式基线

采用 MiniLM 的 384 维文本向量，RQ-VAE 使用三层、每层 256 个码字，潜空间为 32 维；附加一位消歧码。T5 式 Encoder–Decoder 随机初始化，使用最近 20 部电影、1,027 个 Token、有效 batch 256。

模型按 valid NDCG@10 选模。最终 test 使用动态 Trie、beam=20 和 Top-10；三个训练种子为 42、43、44，报告均值与总体标准差（ddof=0）。主实验及基线口径差异见 [Week 5–6 结果](../results/week5-6-generative/06-reports/main_results.md)。

本项目实现语义ID自回归推荐的核心范式；数据、1,027词表和模型规模以仓库协议为准，不将本表视为TIGER原论文指标复现。

## Week 7 融合对照

协同向量只由 train 交互构造，不读取 valid/test 交互。文本 384 维与协同 64 维分别归一化后拼接，再整体归一化；四组 RQ-VAE 的输入维度和码本容量一致。

| 组别 | 协同权重 | 研究问题 |
|---|---|---|
| Text | 0 | 受控纯文本基线 |
| Fixed | 1 | 固定强度的协同融合是否有效 |
| Fixed-Match | 常数，使平均协同平方范数占比匹配 Adaptive | 排除平均强度差异 |
| Adaptive | 由物品 train 频次决定，范围 0–1 | 频次分配是否有额外价值 |

四组量化均使用 seed 42、三层 256 码本；输出ID和质量检查见 [量化结果](../results/week7-improvement/03-semantic-ids/README.md)。不同组的重建目标不同，R² 不用于给推荐效果排名。

四组生成模型 seed42 的验证集筛选已完成，按 valid NDCG@10 冻结 Fixed；控制变量与输入身份记录在 [selection.json](../results/week7-improvement/selection.json)。SASRec / GRU4Rec 的 h20 基线也已各完成三个种子，均按完整历史屏蔽后的 valid NDCG@10 选模。

Text与Fixed的seed43/44已补齐，六个生成模型运行及test解锁检查均通过。Text、Fixed、SASRec和GRU4Rec的12项正式test均已完成，每项覆盖6,040名用户。单seed筛选与三seed正式比较分别报告，test不参与候选选择。

新训练方案使用有效 batch 256、peak lr 3e-4、warmup 1,500、最多 60,000 step 和余弦衰减。旧方案同时改变训练预算和学习率轨迹，因此两者对比不能只归因于训练步数。已执行的10,000次用户配对bootstrap反映当前训练模型条件下的用户抽样不确定性，不能代替更多训练种子；[正式结果](../results/week7-improvement/06-evaluation/main_results.md)同时报告各分组和三种子标准差。

## 证据边界

- 当前只覆盖MovieLens 1M与三个推荐模型训练种子；四组量化器均固定seed42，未单独估计量化随机性。
- Fixed相对GRU4Rec的总体差值配对区间包含0，长尾与少样本组仍较弱；相对Text的提升不代表生成式架构全面优于判别式架构。
- 研究改进是将train-only协同表示融入语义ID，未实现量化器与推荐目标的端到端联合优化。
- 固定预算不保证充分收敛。SVD重建存在浮点差异，原始文件身份以冻结哈希为准，独立重训以新冻结输入和多种子结果评估。
