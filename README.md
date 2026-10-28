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

更新于 **2026-10-28**.

| 周次 | 工作 | 状态 |
|---|---|---|
| 第 1 周 | MovieLens 1M 下载,5-core 预处理,时间切分,统计分析 | ✅ 完成 |
| 第 2 周 | SASRec 三随机种子,GRU4Rec 基线与 checkpoint 复评 | ✅ 完成 |
| 第 3–4 周 | 文本编码,RQ-Kmeans / RQ-VAE,语义 ID 与码本分析 | ✅️已完成 |
| 第 5–6 周 | T5 类生成式推荐:管线验证、三种子训练、约束解码与基线复评 | ✅ 完成 |
| 第 7 周 | 文本—协同融合、语义ID消融与仓库整理 | 进行中：Step 7.0–7.3 |

### 已得到的核心结果

测试集采用 **3,416 个物品全量排序**,主表对所有模型统一屏蔽用户完整历史中的物品.结果为三个随机种子的均值 ± 总体标准差.

| 模型 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|
| SASRec | 0.1976 ± 0.0054 | 0.2832 ± 0.0039 | 0.1361 ± 0.0039 | 0.1637 ± 0.0034 |
| GRU4Rec | **0.2300 ± 0.0019** | **0.3196 ± 0.0007** | **0.1594 ± 0.0016** | **0.1882 ± 0.0009** |
| Generative-T5 | 0.1419 ± 0.0145 | 0.2131 ± 0.0156 | 0.0950 ± 0.0088 | 0.1180 ± 0.0092 |

生成模型的无约束输出没有目录外 ID,但平均 37.69% 的推荐槽位是已见物品;动态 Trie 约束将目录外、已见和短列表率全部降至 0.当前实验中生成模型未超过 GRU4Rec,长尾组也未显示优势.完整主表、分组结果、旧日志复现与解码样例见 [`results/README.md`](results/README.md).

## 实验路线

| 周次 | 输入 | 关键步骤 | 主要输出 |
|---|---|---|---|
| Week 1 | MovieLens 1M 原始评分与电影元数据 | 5-core,ID 重映射,按时间 leave-one-out,无泄漏校验 | `.inter`,三份 `jsonl`,统计表与分布图 |
| Week 2 | 行为序列 | RecBole 复现 SASRec / GRU4Rec,多种子全量排序评测 | 训练日志,checkpoint,基线表 |
| Week 3–4 | `title + year + genres` | MiniLM 文本编码,残差量化,冲突消解,码本质量分析 | `item_emb.npy`,4-token 语义 ID,码本报告 |
| Week 5–6 | 语义 ID 序列 | 审计接口,验证训练链路,选择学习率,三种子训练,约束解码与统一口径复评 | 日志、指标、分组对比与解码样例 |
| Week 7 | 文本向量与训练交互 | SVD协同表示、四组融合、量化与消融 | 已完成448维输入；量化与正式训练待开展 |

Week 7已完成CPU管线验证与四组融合输入，尚无新的推荐指标；[当前产物与验证](results/week7-improvement/README.md)单独记录。

## Week 1–6 冻结实验协议

| 类别 | 设置 |
|---|---|
| 数据 | MovieLens 1M;隐式反馈;5-core 迭代过滤;ID 重映射为连续整数 |
| 划分 | 每个用户按时间排序,最后 1 条为 test,倒数第 2 条为 valid |
| 评测 | 3,416 物品全量排序;Recall@5/10,NDCG@5/10;最终主表统一屏蔽完整历史物品 |
| 基线 | RecBole 1.2.1;CE;最大序列长度 200;以 valid NDCG@10 早停 |
| 文本 | `{title} ({year}) \| {genres}`;`all-MiniLM-L6-v2`;384 维 L2 单位向量 |
| 量化 | RQ-Kmeans baseline;RQ-VAE 主方案;`L=3`,`K=256`,latent=32 |
| 语义 ID | 3 层量化码 + 1 位碰撞消歧码,共 4 token;物品词表大小 1,024 |
| 生成模型 | 随机初始化 T5 式 Encoder–Decoder;11.28M 参数;词表 1,027;历史 20 个物品 |
| 生成训练 | 目标为 4 个位置 token + EOS;AdamW;BF16;有效 batch 256;pilot 选定 lr=`3e-4`;valid NDCG@10 选模 |
| 解码 | beam=20,Top-10;动态 Trie 只允许生成目录内且用户未见的物品 |

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
python scripts/check_prefix_consistency.py --emb item_emb.npy --ids semantic_ids_rqvae.json --text /data/zby/data/ml-1m/processed/item_text.json --outdir results/week3-4-semantic-id/04-prefix-validation
python scripts/export_semantic.py --ids semantic_ids_rqvae.json --emb item_emb.npy --quantizer_checkpoint saved/rqvae_ml1m.pt --processed /data/zby/data/ml-1m/processed --outdir exported

# Week 5–6 Stage 0–1:冻结接口审计与代码静态检查
python -m py_compile scripts/generative_common.py scripts/train_generative.py scripts/evaluate_generative.py scripts/eval_baseline_groups.py scripts/summarize_generative.py
python scripts/train_generative.py --config config/generative_ml1m.yaml --audit_only

# Stage 2:冒烟、断点恢复和小样本过拟合
python scripts/train_generative.py --config config/generative_ml1m.yaml --seed 42 --run_name smoke --max_steps 20 --eval_steps 10 --max_train_examples 2048 --max_eval_users 128 --result_dir results/week5-6-generative/01-pipeline-validation
python scripts/train_generative.py --config config/generative_ml1m.yaml --seed 42 --run_name smoke --max_steps 30 --eval_steps 10 --max_train_examples 2048 --max_eval_users 128 --resume saved/generative/smoke/last --result_dir results/week5-6-generative/01-pipeline-validation
python scripts/train_generative.py --config config/generative_ml1m.yaml --seed 42 --run_name overfit256 --lr 1e-3 --max_steps 1000 --eval_steps 200 --max_train_examples 256 --max_eval_users 128 --result_dir results/week5-6-generative/01-pipeline-validation

# Stage 3:固定 seed 的两档学习率预实验
python scripts/train_generative.py --config config/generative_ml1m.yaml --seed 42 --run_name pilot_lr3e-4 --lr 3e-4 --max_steps 5000 --eval_steps 2500 --max_eval_users 1000 --result_dir results/week5-6-generative/02-learning-rate-selection
python scripts/train_generative.py --config config/generative_ml1m.yaml --seed 42 --run_name pilot_lr1e-3 --lr 1e-3 --max_steps 5000 --eval_steps 2500 --max_eval_users 1000 --result_dir results/week5-6-generative/02-learning-rate-selection
python scripts/summarize_generative.py --results results/week5-6-generative --pilot_only

# Stage 4:三随机种子正式训练
for seed in 42 43 44; do
  python scripts/train_generative.py --config config/generative_ml1m.yaml --seed "$seed" --run_name "t5_sid_seed${seed}" 2>&1 | tee "results/week5-6-generative/03-formal-training/train_seed${seed}.log"
done

# Stage 5:valid/test 的约束与无约束评测
for seed in 42 43 44; do
  python scripts/evaluate_generative.py --config config/generative_ml1m.yaml --checkpoint "saved/generative/t5_sid_seed${seed}/best" --seed "$seed" --split valid --mode both
  python scripts/evaluate_generative.py --config config/generative_ml1m.yaml --checkpoint "saved/generative/t5_sid_seed${seed}/best" --seed "$seed" --split test --mode both --save_predictions
done

# Stage 6:基线统一口径复评（下列命令分别替换模型、seed 与 checkpoint，运行 6 组）
python scripts/eval_baseline_groups.py --model SASRec --checkpoint saved/<checkpoint>.pth --seed 42 --split test

# Stage 7:完整性检查与最终报告
python scripts/summarize_generative.py --results results/week5-6-generative --check_only
python scripts/summarize_generative.py --results results/week5-6-generative --output results/week5-6-generative/06-reports/main_results.md
```

Week 7的CPU准备入口如下，要求现有exported/冻结接口和item_emb.npy；生成的数组留在work/week7/：

```bash
python -m unittest discover -s tests -p "test_week7.py" -v
python scripts/run_week7.py init
python scripts/run_week7.py features
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
│   ├── rqvae_ml1m.yaml
│   ├── generative_ml1m.yaml
│   └── week7/
│       ├── rqvae.yaml
│       ├── generative_historical.yaml
│       ├── generative_text.yaml
│       ├── generative_fixed.yaml
│       ├── generative_fixed_match.yaml
│       └── generative_adaptive.yaml
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
│   ├── export_semantic.py
│   ├── generative_common.py
│   ├── train_generative.py
│   ├── evaluate_generative.py
│   ├── eval_baseline_groups.py
│   ├── summarize_generative.py
│   ├── week7_common.py
│   ├── week7_prepare.py
│   ├── week7_train.py
│   ├── week7_evaluate.py
│   ├── week7_baseline.py
│   ├── week7_report.py
│   ├── week7_release.py
│   └── run_week7.py
├── tests/
│   └── test_week7.py
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
│   │   ├── gru4rec_s42.log
│   │   ├── gru4rec_s43.log
│   │   └── gru4rec_s44.log
│   ├── week3-4-semantic-id/
│   │   ├── README.md
│   │   ├── 01-quantization/
│   │   │   ├── residual_quantization.md
│   │   │   ├── semantic_ids_{kmeans,rqvae}.json
│   │   │   └── {rq_kmeans,rqvae_train}.log
│   │   ├── 02-codebook-analysis/
│   │   │   ├── codebook_report.md
│   │   │   ├── codebook_stats.json
│   │   │   └── figures/（6 张码本利用率图）
│   │   ├── 03-grid-search/
│   │   │   ├── grid_search.md
│   │   │   ├── grid_search.log
│   │   │   ├── candidates/（9 份候选语义 ID）
│   │   │   └── logs/（9 份独立实验日志）
│   │   ├── 04-prefix-validation/
│   │   │   ├── prefix_consistency.md
│   │   │   └── prefix_samples.csv
│   │   └── 05-frozen-interface/
│   │       ├── semantic_ids.json
│   │       ├── train.semantic.jsonl
│   │       ├── valid.semantic.jsonl
│   │       ├── test.semantic.jsonl
│   │       ├── token_vocab.json
│   │       └── meta.json
│   ├── week5-6-generative/
│   │   ├── README.md
│   │   ├── 01-pipeline-validation/（审计、冒烟、恢复、过拟合）
│   │   ├── 02-learning-rate-selection/（两档 pilot 与选择报告）
│   │   ├── 03-formal-training/（三种子日志、轨迹与 best-valid）
│   │   ├── 04-generative-evaluation/（12 份生成评测指标）
│   │   ├── 05-baseline-evaluation/（6 份基线复评指标与日志）
│   │   └── 06-reports/
│   │       ├── main_results.md
│   │       └── decode_samples.md
│   └── week7-improvement/
│       ├── README.md
│       ├── 01-diagnostics/
│       │   ├── environment.json
│       │   ├── input_sha256.txt
│       │   ├── reused_code_sha256.txt
│       │   └── cpu-smoke/
│       │       ├── smoke_historical_s42_best.json
│       │       └── smoke_historical_s42_history.jsonl
│       └── 02-embeddings/
│           ├── meta.json
│           ├── weights.json
│           └── neighbors.json
└── note/
    ├── week0-生成式推荐_三阶段全链路.md
    ├── week1-真实数据实例说明.md
    ├── week2-前置知识-RecBole与基线模型.md
    ├── week3&4-前置知识-向量量化与语义标识符.md
    ├── week5&6-生成式模型超详细解读.md
    └── 按日期记录的实验笔记
```

### 2. 服务器

服务器保留实验运行时的原始落盘结构；GitHub 仓库中的同一批证据按实验阶段重新分组，文件内容未被改写。

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
│   ├── rqvae_ml1m.yaml
│   ├── generative_ml1m.yaml
│   └── week7/
│       ├── rqvae.yaml
│       ├── generative_historical.yaml
│       ├── generative_text.yaml
│       ├── generative_fixed.yaml
│       ├── generative_fixed_match.yaml
│       └── generative_adaptive.yaml
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
│   ├── export_semantic.py
│   ├── generative_common.py
│   ├── train_generative.py
│   ├── evaluate_generative.py
│   ├── eval_baseline_groups.py
│   ├── summarize_generative.py
│   ├── week7_common.py
│   ├── week7_prepare.py
│   ├── week7_train.py
│   ├── week7_evaluate.py
│   ├── week7_baseline.py
│   ├── week7_report.py
│   ├── week7_release.py
│   └── run_week7.py
├── tests/
│   └── test_week7.py
├── log/
│   ├── GRU4Rec/
│   │   ├── GRU4Rec-ml-1m-Oct-15-2026_09-39-08-c5973e.log
│   │   ├── GRU4Rec-ml-1m-Oct-22-2026_18-42-27-15b898.log
│   │   └── GRU4Rec-ml-1m-Oct-22-2026_18-51-07-e50419.log
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
│   ├── gru4rec_s43.log
│   ├── gru4rec_s44.log
│   ├── rq_kmeans.log
│   ├── rqvae_train.log
│   ├── debug/
│   ├── week3-4-semantic-id/
│   │   ├── residual_quantization.md
│   │   ├── grid_search.md
│   │   ├── codebook_report.md
│   │   ├── grid_search.log
│   │   ├── codebook_stats.json
│   │   ├── prefix_consistency.md
│   │   ├── prefix_samples.csv
│   │   ├── figs/
│   │   │   ├── util_kmeans_level0.png
│   │   │   ├── util_kmeans_level1.png
│   │   │   ├── util_kmeans_level2.png
│   │   │   ├── util_rqvae_level0.png
│   │   │   ├── util_rqvae_level1.png
│   │   │   └── util_rqvae_level2.png
│   │   └── grid/
│   │       ├── semantic_ids_kmeans_L2_K64.json
│   │       ├── semantic_ids_kmeans_L2_K256.json
│   │       ├── semantic_ids_kmeans_L3_K64.json
│   │       ├── semantic_ids_kmeans_L3_K256.json
│   │       ├── semantic_ids_rqvae_L2_K64.json
│   │       ├── semantic_ids_rqvae_L2_K256.json
│   │       ├── semantic_ids_rqvae_L3_K64.json
│   │       ├── semantic_ids_rqvae_L3_K256.json
│   │       ├── semantic_ids_rqvae_L4_K256_no_dedup.json
│   │       └── logs/（对应 9 组独立日志）
│   ├── week5-6-generative/
│   │   ├── audit.log
│   │   ├── stage1_static_checks.log
│   │   ├── smoke.log
│   │   ├── smoke_history.jsonl
│   │   ├── smoke_best_valid_metrics.json
│   │   ├── overfit256.log
│   │   ├── overfit256_history.jsonl
│   │   ├── overfit256_best_valid_metrics.json
│   │   ├── pilot_lr3e-4.log
│   │   ├── pilot_lr3e-4_history.jsonl
│   │   ├── pilot_lr3e-4_best_valid_metrics.json
│   │   ├── pilot_lr1e-3.log
│   │   ├── pilot_lr1e-3_history.jsonl
│   │   ├── pilot_lr1e-3_best_valid_metrics.json
│   │   ├── pilot_summary.md
│   │   ├── stage0-3_report.md
│   │   ├── train_seed{42,43,44}.log
│   │   ├── t5_sid_seed{42,43,44}_history.jsonl
│   │   ├── t5_sid_seed{42,43,44}_best_valid_metrics.json
│   │   ├── generative_{valid,test}_{constrained,unconstrained}_seed{42,43,44}_metrics.json
│   │   ├── generative_test_{constrained,unconstrained}_seed{42,43,44}_predictions.jsonl
│   │   ├── baseline_{sasrec,gru4rec}_test_seed{42,43,44}_metrics.json
│   │   ├── baseline_eval.log
│   │   ├── main_results.md
│   │   └── decode_samples.md
│   └── week7-improvement/
│       ├── 01-diagnostics/（输入与旧代码哈希、环境）
│       ├── 02-embeddings/（meta、weights、neighbors）
│       ├── 03-semantic-ids/（预建空目录）
│       ├── 04-screening/（预建空目录）
│       ├── 05-formal-training/（当前仅CPU smoke记录）
│       ├── 06-evaluation/（预建空目录）
│       └── 07-release/（预建空目录）
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
│   ├── GRU4Rec-Oct-22-2026_18-42-50.pth
│   ├── GRU4Rec-Oct-22-2026_18-51-36.pth
│   ├── rq_kmeans_ml1m.npz
│   ├── rqvae_ml1m.pt
│   ├── generative/
│   │   ├── smoke/{best,last}/
│   │   ├── overfit256/{best,last}/
│   │   ├── pilot_lr3e-4/{best,last}/
│   │   ├── pilot_lr1e-3/{best,last}/
│   │   ├── t5_sid_seed42/{best,last}/
│   │   ├── t5_sid_seed43/{best,last}/
│   │   └── t5_sid_seed44/{best,last}/
│   └── week7/
│       └── generative/
│           └── smoke_historical_s42/{best,last}/
├── work/
│   └── week7/
│       ├── collab.npy
│       ├── text.npy
│       ├── fixed.npy
│       ├── fixed_match.npy
│       ├── adaptive.npy
│       └── frequency.json
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
    ├── GRU4Rec-ml-1m-Oct-15-2026_09-39-08-c5973e/
    │   ├── events.out.tfevents.1792028377.amax510.1040029.0
    │   └── 1792031687.831251/events.out.tfevents.1792031687.amax510.1040029.1
    ├── GRU4Rec-ml-1m-Oct-22-2026_18-42-27-15b898/
    │   ├── events.out.tfevents.1792665770.amax510.962347.0
    │   └── 1792666916.7895434/events.out.tfevents.1792666916.amax510.962347.1
    └── GRU4Rec-ml-1m-Oct-22-2026_18-51-07-e50419/
        ├── events.out.tfevents.1792666296.amax510.1007330.0
        └── 1792667590.097698/events.out.tfevents.1792667590.amax510.1007330.1
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

这是一个持续推进的科研仓库:已完成内容均提供真实日志和可复现数字,未运行实验明确标记为待完成,计划值不作为实验结果展示.
