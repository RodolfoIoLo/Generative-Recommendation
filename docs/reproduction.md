# 🚀 复现指南

安装与基础测试从仓库根目录运行。测试只用合成小数据，不需要下载数据集、编码器或模型权重；重训使用下方独立目录，保留仓库中的原始实验记录。

## 1. 安装与 CPU 测试

推荐使用独立的 Python 3.11 环境。先按运行设备安装 PyTorch，再安装项目依赖；以下为 CPU 安装示例：

~~~bash
conda create -n genrec python=3.11 -y
conda activate genrec
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -p "test_week7.py" -v
python -m compileall -q scripts tests
~~~

正式生成模型训练和评测需要 CUDA 版 PyTorch 与可用 GPU。根据机器的驱动与运行环境选择 PyTorch 版本，再安装同一份 `requirements.txt`。实验服务器使用 Python 3.11.16 和 PyTorch 2.14.0+cu126；[服务器环境快照](../environment/server-requirements.txt)记录了完整包版本，供对照和排查。快照包含平台相关依赖，在其他机器上不保证能直接安装。

在与实验服务器相近的 Linux 和 CUDA 环境中，也可在新的独立环境里尝试安装快照：

~~~bash
python -m pip install torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r environment/server-requirements.txt
python -m pip check
~~~

复现实验还需使用相同的数据、模型权重、配置和随机种子；环境一致不保证结果逐位相同。SVD受线性代数后端与线程设置影响，重建向量可能有微小浮点差异。原始SHA-256用于核对冻结文件；独立重训应重新冻结输入，在同一批输入和配置下比较各组，并报告三种子统计。

## 2. 准备数据与文本向量

首次复现先创建独立实验目录；若该目录已有实验，请另取目录名。训练脚本会拒绝覆盖同名结果，直接在附带完成记录的仓库目录重训会触发保护：

~~~bash
set -euo pipefail
mkdir -p work
mkdir work/reproduction
cp -a scripts config work/reproduction/
cp -a results/week3-4-semantic-id/05-frozen-interface work/reproduction/exported
cd work/reproduction
set -euo pipefail
~~~

后续数据准备、训练与评测命令均在这个目录的同一个Bash终端执行，适用于Linux或WSL。错误会立即停止后续步骤；重新打开终端后需恢复环境变量。长训练可放在tmux会话中运行。

从 [GroupLens](https://grouplens.org/datasets/movielens/1m/) 获取 MovieLens 1M，将 `ratings.dat` 与 `movies.dat` 放入数据根目录的 `ml-1m/raw/`，然后预处理：

~~~bash
export GENREC_DATA_ROOT=/path/to/data
python scripts/week1_preprocess.py --dataset ml-1m --data-root "$GENREC_DATA_ROOT"
python scripts/week1_verify_split.py --data-dir "$GENREC_DATA_ROOT/ml-1m/processed"
mkdir -p "$GENREC_DATA_ROOT/recbole"
if [ ! -e "$GENREC_DATA_ROOT/recbole/ml-1m" ]; then
  ln -s "$GENREC_DATA_ROOT/ml-1m/processed" "$GENREC_DATA_ROOT/recbole/ml-1m"
fi
~~~

生成模型需要的六份冻结接口已复制到独立目录的 `exported/`。

Week 7还需要与item ID顺序对应、逐行L2归一化的384维 `item_emb.npy`。以下使用[官方MiniLM模型](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)，首次运行会下载权重；离线环境可将模型标识换成已准备好的模型目录：

~~~bash
python scripts/week3_4_encode_items.py \
  --text "$GENREC_DATA_ROOT/ml-1m/processed/item_text.json" \
  --model sentence-transformers/all-MiniLM-L6-v2 --out item_emb_raw.npy
python -c 'import numpy as np; x=np.load("item_emb_raw.npy"); np.save("item_emb.npy", x/np.linalg.norm(x,axis=1,keepdims=True))'
~~~

精确重现还需要相同的编码器权重、数据及软件环境；输入指纹见 [meta.json](../results/week7-improvement/02-embeddings/meta.json)。数据准备不需要 GPU。

## 3. 构建最终方案的融合输入和语义 ID

~~~bash
python scripts/week7_run.py init
python scripts/week7_run.py features
for variant in text fixed fixed_match adaptive; do
  python scripts/week7_run.py tokenize --variant "$variant"
done
~~~

量化默认使用CPU。结果分别写入 `work/week7/`、`exported/week7/` 和 `saved/week7/`；已有同名量化器会阻止静默覆盖。

## 4. 训练与验证选模

完整链路为：数据处理 → 文本与协同表示 → 语义ID → 序列模型训练 → 验证选模 → test → 统计对比。量化完成后，先训练四组seed42，再仅按valid选择一次候选。卡号按运行设备填写：

~~~bash
GENREC_GPU=0
for variant in text fixed fixed_match adaptive; do
  python scripts/week7_run.py train --variant "$variant" --seed 42 --gpu "$GENREC_GPU"
done
python scripts/week7_run.py select
~~~

Text与selected补齐seed43/44，候选和训练配置保持冻结：

~~~bash
for variant in text selected; do
  for seed in 43 44; do
    python scripts/week7_run.py train --variant "$variant" --seed "$seed" --gpu "$GENREC_GPU"
  done
done
~~~

判别式基线使用相同历史长度20，并按完整历史屏蔽后的valid NDCG@10选模：

~~~bash
for model in SASRec GRU4Rec; do
  for seed in 42 43 44; do
    python scripts/week7_run.py baseline --model "$model" --seed "$seed" \
      --split train --data_path "$GENREC_DATA_ROOT/recbole" --gpu "$GENREC_GPU"
  done
done
~~~

训练中断时，对同一模型和种子追加 `--resume`，例如：

~~~bash
python scripts/week7_run.py train --variant text --seed 43 --gpu "$GENREC_GPU" --resume
~~~

恢复沿用保存的优化器、学习率、随机数和数据位置。新checkpoint完整写入后再替换旧版本；磁盘不足会等待重试。训练记录见 [三种子结果](../results/week7-improvement/05-formal-training/README.md)，选择依据见 [验证集筛选](../results/week7-improvement/04-screening/README.md)。

## 5. 正式评测与结果分析

六个生成模型运行全部完成后，固定beam20、全部6,040名用户，分别评测生成模型和判别式基线：

~~~bash
for variant in text selected; do
  for seed in 42 43 44; do
    python scripts/week7_run.py evaluate --variant "$variant" --seed "$seed" \
      --split test --gpu "$GENREC_GPU"
  done
done
for model in SASRec GRU4Rec; do
  for seed in 42 43 44; do
    python scripts/week7_run.py baseline --model "$model" --seed "$seed" \
      --split test --data_path "$GENREC_DATA_ROOT/recbole" --gpu "$GENREC_GPU"
  done
done
python scripts/week7_run.py curves
python scripts/week7_run.py summarize
~~~

已有的判别式基线checkpoint也可在CPU按相同口径复评：将对应 `baseline --split test` 命令的 `--gpu "$GENREC_GPU"` 换为 `--cpu`。训练及生成模型正式评测仍使用GPU。

汇总输出写入`results/week7-improvement/06-evaluation/`：`summaries/`保存三种子和配对统计，`figures/`保存对比图，`main_results.md`连接这些证据。完整预测位于 `work/week7/predictions/`；仓库保留指标、曲线和固定20名用户样例。test不参与候选选择，完整口径见 [实验协议](experiment-protocol.md)，[已完成的结果](../results/week7-improvement/06-evaluation/main_results.md)提供对照。


## 6. 回看Week 1–6的完整路线

上面的主路线重建最终Week 7实验。以下命令用于逐周学习早期实验，仍在独立实验目录执行；先完成数据预处理、划分检查和文本编码。模型checkpoint名称由训练时生成，基线复评需传入对应日志记录的实际路径。

### 数据分析与旧基线

~~~bash
python scripts/week1_stats.py --dataset ml-1m --data-root "$GENREC_DATA_ROOT"
for model in SASRec GRU4Rec; do
  for seed in 42 43 44; do
    python scripts/week2_train_baseline.py --model "$model" --seed "$seed" \
      --data_path "$GENREC_DATA_ROOT/recbole"
  done
done
~~~

旧基线读取`config/sasrec_ml1m.yaml`，历史上限为200；最终h20基线由Week 7入口单独训练。日志和权重保存在运行目录；可用`week2_evaluate_checkpoint.py --help`查看补评参数。

### 文本量化与冻结接口

若要从头重建Week 3–4接口，先为本次运行另建导出目录；以下会生成新的量化权重和语义ID，不覆盖附带的冻结接口：

~~~bash
mkdir -p results/week3-4-semantic-id/01-quantization/logs
python scripts/week3_4_build_semantic_ids.py --config config/rqvae_ml1m.yaml \
  --emb item_emb.npy --method kmeans --checkpoint saved/rq_kmeans_rebuilt.npz \
  --out results/week3-4-semantic-id/01-quantization/identifiers/semantic_ids_kmeans.json
python scripts/week3_4_build_semantic_ids.py --config config/rqvae_ml1m.yaml \
  --emb item_emb.npy --method rqvae --device cpu --checkpoint saved/rqvae_rebuilt.pt \
  --out results/week3-4-semantic-id/01-quantization/identifiers/semantic_ids_rqvae.json
python scripts/week3_4_analyze_codebook.py \
  --kmeans results/week3-4-semantic-id/01-quantization/identifiers/semantic_ids_kmeans.json \
  --rqvae results/week3-4-semantic-id/01-quantization/identifiers/semantic_ids_rqvae.json
python scripts/week3_4_check_prefix.py \
  --ids results/week3-4-semantic-id/01-quantization/identifiers/semantic_ids_rqvae.json \
  --emb item_emb.npy --text "$GENREC_DATA_ROOT/ml-1m/processed/item_text.json"
python scripts/week3_4_export_semantic.py \
  --ids results/week3-4-semantic-id/01-quantization/identifiers/semantic_ids_rqvae.json \
  --processed "$GENREC_DATA_ROOT/ml-1m/processed" --outdir exported-rebuilt \
  --emb item_emb.npy --quantizer_checkpoint saved/rqvae_rebuilt.pt \
  --encoder sentence-transformers/all-MiniLM-L6-v2
~~~

重新导出的六份文件必须作为一套使用，不可把新ID混入旧序列。主路线使用随仓库复制的`exported/`，因此不会将新旧量化结果混用。网格实验改变`--n_levels`和`--codebook_size`，完整已运行组合与用途见[网格报告](../results/week3-4-semantic-id/03-grid-search/grid_search.md)。

### 首版生成模型

~~~bash
python scripts/week5_6_train.py --config config/generative_ml1m.yaml --audit_only
for seed in 42 43 44; do
  python scripts/week5_6_train.py --config config/generative_ml1m.yaml --seed "$seed"
  python scripts/week5_6_evaluate.py --config config/generative_ml1m.yaml \
    --checkpoint "saved/generative/t5_sid_seed$seed/best" --seed "$seed" --split valid --mode both
  python scripts/week5_6_evaluate.py --config config/generative_ml1m.yaml \
    --checkpoint "saved/generative/t5_sid_seed$seed/best" --seed "$seed" --split test --mode both --save_predictions
done
~~~

使用CUDA版PyTorch和可用GPU运行训练／评测；可在命令前用`CUDA_VISIBLE_DEVICES=卡号`选择设备。首版预算30k、线性学习率衰减，与最终60k余弦方案分别报告。

生成汇总前，还要加载两种旧基线各自三个种子的checkpoint进行完整历史屏蔽复评。以下替换为日志中记录的**真实权重路径和对应种子**，对六个checkpoint各执行一次，然后汇总：

~~~bash
python scripts/week5_6_evaluate_baselines.py --model SASRec --seed 42 \
  --checkpoint /path/to/SASRec-seed42.pth --split test
# 以相同方式复评SASRec seed43/44与GRU4Rec seed42/43/44。
python scripts/week5_6_report.py --results results/week5-6-generative
~~~

独立重训不要求碰巧复现原项目日志的小数值。只有核验原项目的冻结checkpoint时才追加`--check_published_logs`，此时脚本会要求旧口径指标与原日志误差不超过`1e-4`；公开的原项目复评文件均已通过该检查。

## 7. 重绘文档中的五张全景图

在仓库根目录用CPU执行`python docs/assets/workflows/draw_workflows.py`，需要Pillow与可用的中文字体。脚本会自动查找常用中文字体，也可用`--font /path/to/font.ttc`指定，输出高清PNG和可无限放大的SVG。图中的文件路径与当前归档结构一致。
