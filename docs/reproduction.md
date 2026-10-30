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

复现实验还需使用相同的数据、模型权重、配置和随机种子；环境一致不保证结果逐位相同。

## 2. 准备数据与文本向量

首次复现先创建独立实验目录；若该目录已有实验，请另取目录名。训练脚本会拒绝覆盖同名结果，直接在附带完成记录的仓库目录重训会触发保护：

~~~bash
mkdir -p work/reproduction
cp -a scripts config work/reproduction/
cp -a results/week3-4-semantic-id/05-frozen-interface work/reproduction/exported
cd work/reproduction
~~~

后续数据准备、训练与评测命令均在这个目录执行。以下命令采用 Bash，适用于 Linux 或 WSL。

从 [GroupLens](https://grouplens.org/datasets/movielens/1m/) 获取 MovieLens 1M，将 `ratings.dat` 与 `movies.dat` 放入数据根目录的 `ml-1m/raw/`，然后预处理：

~~~bash
export GENREC_DATA_ROOT=/path/to/data
python scripts/preprocess.py --dataset ml-1m --data-root "$GENREC_DATA_ROOT"
mkdir -p "$GENREC_DATA_ROOT/recbole"
ln -s "$GENREC_DATA_ROOT/ml-1m/processed" "$GENREC_DATA_ROOT/recbole/ml-1m"
~~~

生成模型需要的六份冻结接口已复制到独立目录的 `exported/`。

Week 7还需要与 item ID 顺序对应、逐行L2归一化的384维 `item_emb.npy`。用已准备的 MiniLM 模型目录编码后归一化：

~~~bash
python scripts/encode_items.py \
  --text "$GENREC_DATA_ROOT/ml-1m/processed/item_text.json" \
  --model /path/to/all-MiniLM-L6-v2 --out item_emb_raw.npy
python -c 'import numpy as np; x=np.load("item_emb_raw.npy"); np.save("item_emb.npy", x/np.linalg.norm(x,axis=1,keepdims=True))'
~~~

精确重现还需要相同的编码器权重、数据及软件环境；输入指纹见 [meta.json](../results/week7-improvement/02-embeddings/meta.json)。数据准备不需要 GPU。

## 3. 构建融合输入和语义 ID

~~~bash
python scripts/run_week7.py init
python scripts/run_week7.py features
for variant in text fixed fixed_match adaptive; do
  python scripts/run_week7.py tokenize --variant "$variant" || break
done
~~~

量化默认使用CPU。结果分别写入 `work/week7/`、`exported/week7/` 和 `saved/week7/`；已有同名量化器会阻止静默覆盖。

## 4. 训练与验证选模

完整链路为：数据处理 → 文本与协同表示 → 语义ID → 序列模型训练 → 验证选模 → test → 统计对比。量化完成后，先训练四组seed42，再仅按valid选择一次候选。卡号按运行设备填写：

~~~bash
GENREC_GPU=0
for variant in text fixed fixed_match adaptive; do
  python scripts/run_week7.py train --variant "$variant" --seed 42 --gpu "$GENREC_GPU"
done
python scripts/run_week7.py select
~~~

Text与selected补齐seed43/44，候选和训练配置保持冻结：

~~~bash
for variant in text selected; do
  for seed in 43 44; do
    python scripts/run_week7.py train --variant "$variant" --seed "$seed" --gpu "$GENREC_GPU"
  done
done
~~~

判别式基线使用相同历史长度20，并按完整历史屏蔽后的valid NDCG@10选模：

~~~bash
for model in SASRec GRU4Rec; do
  for seed in 42 43 44; do
    python scripts/run_week7.py baseline --model "$model" --seed "$seed" \
      --split train --data_path "$GENREC_DATA_ROOT/recbole" --gpu "$GENREC_GPU"
  done
done
~~~

训练中断时，对同一模型和种子追加 `--resume`，例如：

~~~bash
python scripts/run_week7.py train --variant text --seed 43 --gpu "$GENREC_GPU" --resume
~~~

恢复沿用保存的优化器、学习率、随机数和数据位置。新checkpoint完整写入后再替换旧版本；磁盘不足会等待重试。训练记录见 [三种子结果](../results/week7-improvement/05-formal-training/README.md)，选择依据见 [验证集筛选](../results/week7-improvement/04-screening/README.md)。

## 5. 正式评测与结果分析

六个生成模型运行全部完成后，固定beam20、全部6,040名用户，分别评测生成模型和判别式基线：

~~~bash
for variant in text selected; do
  for seed in 42 43 44; do
    python scripts/run_week7.py evaluate --variant "$variant" --seed "$seed" \
      --split test --gpu "$GENREC_GPU"
  done
done
for model in SASRec GRU4Rec; do
  for seed in 42 43 44; do
    python scripts/run_week7.py baseline --model "$model" --seed "$seed" \
      --split test --data_path "$GENREC_DATA_ROOT/recbole" --gpu "$GENREC_GPU"
  done
done
python scripts/run_week7.py curves
python scripts/run_week7.py summarize
~~~

汇总输出三种子主表、分组指标和用户配对bootstrap。完整预测位于 `work/week7/predictions/`；仓库保留指标、曲线和固定20名用户样例。test不参与候选选择，完整口径见 [实验协议](experiment-protocol.md)。
