# GENREC-demo:基于语义标识符的生成式推荐

> 7 周科研轮转项目,持续更新中.目标是从可复现的序列推荐基线出发,构建"物品文本 → 语义标识符 → 自回归推荐"的完整实验链路.

## 这是什么

传统推荐通常为候选物品逐个打分,再从数千个分数中排序;生成式推荐则先把物品表示成一串离散语义 token,让模型像生成句子一样,直接生成下一个物品的标识符.作为端到端的推荐新范式, 生成式推荐可以更好地利用物品文本,用户序列和语义结构,尤其在冷启动和长尾场景中,是未来推荐系统的一个重要方向.

```text
传统推荐:用户历史 → 为 3,416 个候选物品打分 → Top-K
生成式推荐:物品文本 → 分层语义 ID → 用户 token 序列 → 自回归生成下一个语义 ID

示意(非当前实验结果):
Toy Story → [37, 128, 5, 0]
```

这种表示把"推荐哪个物品"转化为"生成哪组语义 token".共享前缀可以表达由粗到细的语义结构,也为统一检索,推荐与新物品建模提供了一个清晰接口.

本仓库强调三件事:

- **可复现**:数据划分,评测协议,随机种子,日志与环境版本均留痕.
- **可比较**:先建立 SASRec / GRU4Rec 全量排序基线,再进入生成式模型.
- **可解释**:不仅报告推荐指标,也检查码本利用率,冲突率和前缀语义一致性.

## 当前进度

更新于 **2026-10-21**.

| 周次 | 工作 | 状态 |
|---|---|---|
| 第 1 周 | MovieLens 1M 下载,5-core 预处理,时间切分,统计分析 | ✅ 完成 |
| 第 2 周 | SASRec 三随机种子,GRU4Rec 基线与 checkpoint 复评 | ✅ 完成 |
| 第 3–4 周 | 文本编码,RQ-Kmeans / RQ-VAE,语义 ID 与码本分析 | ✅️已完成 |
| 第 5–6 周 | T5 类生成式推荐训练,全量排序评测与基线对比 | ⏳ 计划中 |
| 第 7 周 | 冷启动 / 长尾改进,消融实验与结项报告 | ⏳ 计划中 |

### 已得到的核心结果

测试集采用 **3,416 个物品全量排序**,不是"1 个正样本 + 少量随机负样本"的采样评测.

| 模型 | Seed | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| SASRec | 42 | 0.1409 | 0.2204 | 0.0884 | 0.1141 |
| SASRec | 43 | 0.1523 | 0.2339 | 0.0968 | 0.1230 |
| SASRec | 44 | 0.1526 | 0.2250 | 0.0958 | 0.1191 |
| **SASRec** | **均值 ± std** | **0.1486 ± 0.0054** | **0.2264 ± 0.0056** | **0.0937 ± 0.0037** | **0.1187 ± 0.0036** |
| **GRU4Rec** | **42** | **0.1861** | **0.2657** | **0.1261** | **0.1517** |

完整产物,数据规格与文件来源见 [`results/README.md`](results/README.md).

## 实验路线

| 周次 | 输入 | 关键步骤 | 主要输出 |
|---|---|---|---|
| Week 1 | MovieLens 1M 原始评分与电影元数据 | 5-core,ID 重映射,按时间 leave-one-out,无泄漏校验 | `.inter`,三份 `jsonl`,统计表与分布图 |
| Week 2 | 行为序列 | RecBole 复现 SASRec / GRU4Rec,多种子全量排序评测 | 训练日志,checkpoint,基线表 |
| Week 3–4 | `title + year + genres` | MiniLM 文本编码,残差量化,冲突消解,码本质量分析 | `item_emb.npy`,4-token 语义 ID,码本报告 |
| Week 5–6 | 语义 ID 序列 | 训练 encoder-decoder Transformer,自回归生成下一物品 | 生成式推荐结果与基线对比 |
| Week 7 | 完整模型与物品频次 | 冷启动 / 长尾实验,消融与误差分析 | 最终报告与可复现实验表 |

## 冻结实验协议

| 类别 | 设置 |
|---|---|
| 数据 | MovieLens 1M;隐式反馈;5-core 迭代过滤;ID 重映射为连续整数 |
| 划分 | 每个用户按时间排序,最后 1 条为 test,倒数第 2 条为 valid |
| 评测 | 全量排序;Recall@5/10,NDCG@5/10;屏蔽训练期已见物品 |
| 基线 | RecBole 1.2.1;CE;最大序列长度 200;以 valid NDCG@10 早停 |
| 文本 | `{title} ({year}) \| {genres}`;`all-MiniLM-L6-v2`;384 维 L2 单位向量 |
| 量化 | RQ-Kmeans baseline;RQ-VAE 主方案;`L=3`,`K=256`,latent=32 |
| 语义 ID | 3 层量化码 + 1 位碰撞消歧码,共 4 token;物品词表大小 1,024 |

## 环境与复现

- Ubuntu 服务器:4 × NVIDIA RTX 4090
- Python 3.11.16
- PyTorch 2.14.0+cu126
- RecBole 1.2.1
- NumPy 1.26.4 / SciPy 1.17.1 / scikit-learn 1.9.1
- 完整环境快照:[`requirements.txt`](requirements.txt)

```bash
# 环境
pip install torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# Week 1:预处理,统计与划分校验
python scripts/preprocess.py --dataset ml-1m
python scripts/stats.py --dataset ml-1m
python scripts/verify_split.py --data-dir /data/zby/data/ml-1m/processed

# Week 2:序列推荐基线
python scripts/run_baseline.py --model SASRec --config config/sasrec_ml1m.yaml --seed 42
python scripts/run_baseline.py --model GRU4Rec --config config/sasrec_ml1m.yaml --seed 42

# Week 3–4:文本编码、语义 ID、前缀验证与接口冻结
python scripts/encode_items.py --model /data/LLMs/all-MiniLM-L6-v2 --out item_emb.npy
python scripts/build_semantic_id.py --config config/rqvae_ml1m.yaml --emb item_emb.npy --method rqvae --device cuda:0 --checkpoint saved/rqvae_ml1m.pt --out semantic_ids_rqvae.json
python scripts/check_prefix_consistency.py --emb item_emb.npy --ids semantic_ids_rqvae.json --text /data/zby/data/ml-1m/processed/item_text.json --outdir results/week3-4-semantic-id
python scripts/export_semantic.py --ids semantic_ids_rqvae.json --emb item_emb.npy --quantizer_checkpoint saved/rqvae_ml1m.pt --processed /data/zby/data/ml-1m/processed --outdir exported
```

MovieLens 1M 可从 [GroupLens](https://grouplens.org/datasets/movielens/1m/) 获取.原始数据,embedding 和 checkpoint 不进入 Git;它们的路径,规格和校验信息记录在结果索引中.

## 目录

### 1. GitHub 仓库

```text
repository/
├── .gitignore
├── README.md
├── requirements.txt
├── config/
│   ├── sasrec_ml1m.yaml
│   └── rqvae_ml1m.yaml
├── scripts/
│   ├── preprocess.py
│   ├── stats.py
│   ├── verify_split.py
│   ├── run_baseline.py
│   ├── eval_only.py
│   ├── export_item_freq.py
│   ├── encode_items.py
│   ├── build_semantic_id.py
│   ├── analyze_codebook.py
│   ├── check_prefix_consistency.py
│   └── export_semantic.py
├── results/
│   ├── README.md
│   ├── week1-data-stats/
│   │   ├── preprocess_log.json
│   │   ├── stats.json
│   │   ├── stats.md
│   │   ├── item_freq.json
│   │   └── figs/
│   │       ├── item_freq_hist.png
│   │       ├── item_freq_loglog.png
│   │       └── seq_len_hist.png
│   ├── week2-sasrec-baseline/
│   │   ├── baseline_table.md
│   │   ├── sasrec_s42.log
│   │   ├── sasrec_s42_patch_test.log
│   │   ├── sasrec_s43.log
│   │   ├── sasrec_s44.log
│   │   └── gru4rec_s42.log
│   └── week3-4-semantic-id/
│       ├── residual_quantization.md
│       ├── grid_search.md
│       ├── codebook_report.md
│       ├── codebook_stats.json
│       ├── semantic_ids_kmeans.json
│       ├── semantic_ids_rqvae.json
│       ├── rq_kmeans.log
│       ├── rqvae_train.log
│       ├── grid_search.log
│       ├── prefix_consistency.md
│       ├── prefix_samples.csv
│       ├── frozen/
│       │   ├── semantic_ids.json
│       │   ├── train.semantic.jsonl
│       │   ├── valid.semantic.jsonl
│       │   ├── test.semantic.jsonl
│       │   ├── token_vocab.json
│       │   └── meta.json
│       ├── figs/
│       │   ├── util_kmeans_level0.png
│       │   ├── util_kmeans_level1.png
│       │   ├── util_kmeans_level2.png
│       │   ├── util_rqvae_level0.png
│       │   ├── util_rqvae_level1.png
│       │   └── util_rqvae_level2.png
│       └── grid/
│           ├── semantic_ids_kmeans_L2_K64.json
│           ├── semantic_ids_kmeans_L2_K256.json
│           ├── semantic_ids_kmeans_L3_K64.json
│           ├── semantic_ids_kmeans_L3_K256.json
│           ├── semantic_ids_rqvae_L2_K64.json
│           ├── semantic_ids_rqvae_L2_K256.json
│           ├── semantic_ids_rqvae_L3_K64.json
│           ├── semantic_ids_rqvae_L3_K256.json
│           ├── semantic_ids_rqvae_L4_K256_no_dedup.json
│           └── logs/
│               ├── kmeans_L2_K64.log
│               ├── kmeans_L2_K256.log
│               ├── kmeans_L3_K64.log
│               ├── kmeans_L3_K256.log
│               ├── rqvae_L2_K64.log
│               ├── rqvae_L2_K256.log
│               ├── rqvae_L3_K64.log
│               ├── rqvae_L3_K256.log
│               └── rqvae_L4_K256_no_dedup.log
└── note/
    ├── week0-生成式推荐_三阶段全链路.md
    ├── week1-真实数据实例说明.md
    ├── week2-前置知识-RecBole与基线模型.md
    ├── week3-前置知识-向量量化与语义标识符.md
    └── 按日期记录的实验笔记
```

### 2. 服务器

<details>
<summary><code>/home/zby/genrec/</code>:代码,训练日志与模型产物</summary>

```text
/home/zby/genrec/
├── requirements.txt
├── item_freq.json
├── item_emb.npy
├── semantic_ids_kmeans.json
├── semantic_ids_rqvae.json
├── config/
│   ├── sasrec_ml1m.yaml
│   └── rqvae_ml1m.yaml
├── scripts/
│   ├── preprocess.py
│   ├── stats.py
│   ├── verify_split.py
│   ├── run_baseline.py
│   ├── eval_only.py
│   ├── export_item_freq.py
│   ├── encode_items.py
│   ├── build_semantic_id.py
│   ├── analyze_codebook.py
│   ├── check_prefix_consistency.py
│   └── export_semantic.py
├── log/
│   ├── GRU4Rec/
│   │   └── GRU4Rec-ml-1m-Oct-15-2026_09-39-08-c5973e.log
│   └── SASRec/
│       ├── SASRec-ml-1m-Sep-28-2026_19-17-25-dbfc42.log
│       ├── SASRec-ml-1m-Oct-10-2026_15-55-25-dbfc42.log
│       ├── SASRec-ml-1m-Oct-15-2026_13-00-11-6e006f.log
│       └── SASRec-ml-1m-Oct-15-2026_17-48-15-37672d.log
├── results/
│   ├── sasrec_s42.log
│   ├── sasrec_s42_patch_test.log
│   ├── sasrec_s43.log
│   ├── sasrec_s44.log
│   ├── gru4rec_s42.log
│   ├── rq_kmeans.log
│   ├── rqvae_train.log
│   ├── debug/
│   └── week3-4-semantic-id/
│       ├── residual_quantization.md
│       ├── grid_search.md
│       ├── codebook_report.md
│       ├── grid_search.log
│       ├── codebook_stats.json
│       ├── prefix_consistency.md
│       ├── prefix_samples.csv
│       ├── figs/
│       │   ├── util_kmeans_level0.png
│       │   ├── util_kmeans_level1.png
│       │   ├── util_kmeans_level2.png
│       │   ├── util_rqvae_level0.png
│       │   ├── util_rqvae_level1.png
│       │   └── util_rqvae_level2.png
│       └── grid/
│           ├── semantic_ids_kmeans_L2_K64.json
│           ├── semantic_ids_kmeans_L2_K256.json
│           ├── semantic_ids_kmeans_L3_K64.json
│           ├── semantic_ids_kmeans_L3_K256.json
│           ├── semantic_ids_rqvae_L2_K64.json
│           ├── semantic_ids_rqvae_L2_K256.json
│           ├── semantic_ids_rqvae_L3_K64.json
│           ├── semantic_ids_rqvae_L3_K256.json
│           ├── semantic_ids_rqvae_L4_K256_no_dedup.json
│           └── logs/（对应 9 组独立日志）
├── exported/
│   ├── semantic_ids.json
│   ├── train.semantic.jsonl
│   ├── valid.semantic.jsonl
│   ├── test.semantic.jsonl
│   ├── token_vocab.json
│   └── meta.json
├── saved/
│   ├── SASRec-Sep-28-2026_19-18-04.pth
│   ├── SASRec-Oct-15-2026_13-00-52.pth
│   ├── SASRec-Oct-15-2026_17-48-40.pth
│   ├── GRU4Rec-Oct-15-2026_09-39-37.pth
│   ├── rq_kmeans_ml1m.npz
│   └── rqvae_ml1m.pt
└── log_tensorboard/
    ├── SASRec-ml-1m-Sep-28-2026_19-17-25-dbfc42/
    │   ├── events.out.tfevents.1790594284.amax510.2957982.0
    │   └── 1790599116.9876099/events.out.tfevents.1790599116.amax510.2957982.1
    ├── SASRec-ml-1m-Oct-10-2026_15-55-25-dbfc42/
    │   └── events.out.tfevents.1791618952.amax510.3426306.0
    ├── SASRec-ml-1m-Oct-15-2026_13-00-11-6e006f/
    │   ├── events.out.tfevents.1792040452.amax510.1638753.0
    │   └── 1792057665.8991668/events.out.tfevents.1792057665.amax510.1638753.1
    ├── SASRec-ml-1m-Oct-15-2026_17-48-15-37672d/
    │   ├── events.out.tfevents.1792057720.amax510.2509236.0
    │   └── 1792069370.1500692/events.out.tfevents.1792069370.amax510.2509236.1
    └── GRU4Rec-ml-1m-Oct-15-2026_09-39-08-c5973e/
        ├── events.out.tfevents.1792028377.amax510.1040029.0
        └── 1792031687.831251/events.out.tfevents.1792031687.amax510.1040029.1
```

#### 训练产物为什么叫这些名字

`results/` 是运行命令时手工给 `tee` 指定的文件名,不遵循 RecBole 的自动命名规则;其余三类由 RecBole / TensorBoard 自动生成:

| 位置 | 命名格式 | 各字段含义 |
|---|---|---|
| `log/<Model>/` | `<Model>-<Dataset>-<启动时间>-<config_hash>.log` | 末尾 6 位是最终配置(包含 seed 等参数)的 MD5 短指纹,用于区分不同实验配置 |
| `saved/` | `<Model>-<Trainer初始化时间>.pth` | 训练期间每次 valid 指标刷新都会覆盖这一个路径,最终留下 best checkpoint.文件名不含数据集、seed 或配置指纹,复现时要结合对应日志或 checkpoint 内的 `config` 判断 |
| `log_tensorboard/` 外层目录 | 与 `log/<Model>/` 的日志主文件名相同 | 因而可以按模型、数据集、启动时间和配置指纹与训练日志一一对应 |
| `events.out.tfevents.<time>.<host>.<pid>.<n>` | TensorBoard 标准事件名 | `time` 是 Unix 时间戳,`host` 是主机名,`pid` 是写入进程号,末位是同一进程的事件文件序号 |
| TensorBoard 内的浮点数子目录 | `<time.time()>/events.out.tfevents...` | RecBole 在训练收尾调用 `add_hparams()` 时创建,保存超参数与 best valid score;这串小数只是 Unix 时间,不是实验指标 |

这些时间都表示相应对象的**创建/初始化时间**,不一定等于训练完成时间或 best epoch 的时间.

</details>

<details>
<summary><code>/data/zby/data/</code>:原始数据与可复现预处理结果</summary>

```text
/data/zby/data/
├── ml-1m/
│   ├── raw/
│   │   ├── ml-1m.zip
│   │   ├── README
│   │   ├── ratings.dat
│   │   ├── movies.dat
│   │   └── users.dat
│   └── processed/
│       ├── ml-1m.inter
│       ├── train.jsonl
│       ├── valid.jsonl
│       ├── test.jsonl
│       ├── user_id_map.json
│       ├── item_id_map.json
│       ├── item_text.json
│       ├── preprocess_log.json
│       └── stats/
│           ├── stats.md
│           ├── stats.json
│           └── figs/
│               ├── item_freq_hist.png
│               ├── item_freq_loglog.png
│               └── seq_len_hist.png
└── recbole/
    └── ml-1m -> /data/zby/data/ml-1m/processed
```

</details>

文本编码器从服务器共享只读目录 `/data/LLMs/all-MiniLM-L6-v2` 加载;`/data/LLMs/bge-m3` 预留作对照,不需要联网下载模型.

## 说明

这是一个仍在推进的科研仓库:已完成内容给出真实日志和可复现数字,尚未运行的实验明确标为待完成,不用计划值冒充结果.
