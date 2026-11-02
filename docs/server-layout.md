# 🗂️ 实验工作区与仓库如何对应

公开仓库保存代码、配置、可阅读的结果和轻量证据；实验工作区保存运行这些程序所需的数据、向量和大型权重。两处`scripts/`与`results/`使用同一命名和分层规则，服务器额外保留完整预测及训练状态。

## 实验工作区

```text
/home/zby/genrec/
├── scripts/                   # 按week前缀排列；公开实现与仓库同步
├── config/                    # 原始与Week 7冻结配置
├── tests/                     # CPU检查
├── results/                   # 与仓库同层级，另有不入库的完整轨迹/预测
│   ├── week1-data-stats/
│   ├── week2-sasrec-baseline/
│   │   ├── training-logs/{sasrec,gru4rec}/
│   │   └── evaluation-logs/sasrec/
│   ├── week3-4-semantic-id/
│   │   ├── 01-quantization/{identifiers,logs}/
│   │   ├── 02-codebook-analysis/figures/{kmeans,rqvae}/
│   │   ├── 03-grid-search/{candidates,logs}/{kmeans,rqvae}/
│   │   ├── 04-prefix-validation/
│   │   └── 05-frozen-interface/
│   ├── week5-6-generative/
│   │   ├── 01-pipeline-validation/{audit,smoke,overfit}/
│   │   ├── 02-learning-rate-selection/{lr3e-4,lr1e-3}/
│   │   ├── 03-formal-training/seed{42,43,44}/
│   │   ├── 04-generative-evaluation/{valid,test}/{constrained,unconstrained}/
│   │   ├── 05-baseline-evaluation/{sasrec,gru4rec}/
│   │   └── 06-reports/
│   └── week7-improvement/
│       ├── selection.json
│       ├── 01-diagnostics/{environment,training-curves,search,cpu-smoke}/
│       ├── 02-embeddings/
│       ├── 03-semantic-ids/方案/{identifiers,quality,training,figures}/
│       ├── 04-screening/training-logs/
│       ├── 05-formal-training/{generative,baselines}/模型或方案/seed/
│       └── 06-evaluation/
│           ├── generative/方案/seed/
│           ├── baselines/模型/seed/
│           ├── summaries/
│           ├── figures/
│           └── main_results.md
├── item_emb.npy               # 3416×384，逐行归一化的文本向量
├── exported/                  # Week 3–4冻结的六份训练接口
│   └── week7/方案/            # 四组表示各自的六份接口
├── saved/
│   ├── SASRec-*.pth / GRU4Rec-*.pth
│   ├── rq_kmeans_ml1m.npz / rqvae_ml1m.pt
│   ├── generative/运行名/{best,last}/
│   └── week7/
│       ├── quantizers/方案.pt
│       ├── generative/运行名/{best,last}/
│       └── baselines/模型_h20_s种子/
├── work/week7/                # 协同/融合向量、频次与完整预测
│   ├── collab.npy / 各方案.npy
│   ├── frequency.json / semantic_ids_方案.json
│   └── predictions/*.jsonl
├── log/模型/                  # RecBole原始日志
└── log_tensorboard/           # TensorBoard事件
```

`best/`用于按valid选出的正式评测，`last/`用于恢复训练；两者不要求处于相同步数。运维脚本和排队记录只保留在实验工作区，不属于公开学习链路。

## 数据与文本编码器

```text
/data/zby/data/
├── ml-1m/
│   ├── raw/                   # ratings.dat、movies.dat等原始文件
│   └── processed/
│       ├── ml-1m.inter
│       ├── train.jsonl / valid.jsonl / test.jsonl
│       ├── user_id_map.json / item_id_map.json / item_text.json
│       ├── preprocess_log.json
│       └── stats/             # stats.json、stats.md和figs/
└── recbole/ml-1m -> ../ml-1m/processed

/data/LLMs/all-MiniLM-L6-v2/   # 本实验读取的现成文本编码器
```

这些路径是本次实验实例；复现可换成自己的数据根目录和编码器位置。公开的六份冻结接口位于[`results/week3-4-semantic-id/05-frozen-interface/`](../results/week3-4-semantic-id/05-frozen-interface/)，训练时对应工作区的`exported/`。

## 自动生成的名字怎么看？

| 位置 | 命名与含义 |
|---|---|
| `log/模型/` | `模型-数据集-启动时间-配置短指纹.log`；短指纹区分配置 |
| `saved/`中的RecBole权重 | `模型-初始化时间.pth`；seed以checkpoint配置和对应日志为准 |
| `log_tensorboard/`外层 | 与同次RecBole运行的日志名对应 |
| `events.out.tfevents.*` | TensorBoard标准事件名，包含时间、主机、进程与序号 |
| TensorBoard浮点数子目录 | 记录超参数摘要时创建的Unix时间，不是实验指标 |

原始日志和冻结JSON中保留运行时的路径与代码名，以保留来源记录。当前脚本通过`week1_paths.py`解析已归档的结果位置；数据接口和checkpoint路径保持稳定，配置签名不因展示目录调整而改变。
