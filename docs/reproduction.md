# 🚀 复现指南

从仓库根目录运行。基础测试只用合成小数据，不需要下载数据集、编码器或模型权重。

## 1. 安装与 CPU 测试

推荐 Python 3.11。先按运行设备安装 PyTorch，再安装项目依赖；以下为 CPU 安装示例：

~~~bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-week7.txt
python -m unittest discover -s tests -p "test_week7.py" -v
python -m compileall -q scripts tests
~~~

正式生成模型训练和评测需要 CUDA 版 PyTorch 与可用 GPU。仓库 [requirements.txt](../requirements.txt) 是实验服务器的完整环境快照；其中的 CUDA 包和本机路径不一定适合其他机器。

## 2. 准备数据与文本向量

从 [GroupLens](https://grouplens.org/datasets/movielens/1m/) 获取 MovieLens 1M，将 `ratings.dat` 与 `movies.dat` 放入数据根目录的 `ml-1m/raw/`，然后预处理：

~~~bash
export GENREC_DATA_ROOT=/path/to/data
python scripts/preprocess.py --dataset ml-1m --data-root "$GENREC_DATA_ROOT"
~~~

生成模型需要原有六份冻结接口，可直接从仓库复制：

~~~bash
cp -a results/week3-4-semantic-id/05-frozen-interface exported
~~~

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

## 4. 后续训练与评测

以下为训练入口示例，卡号按实际空闲设备指定。对应变体的量化接口须先完成：

~~~bash
python scripts/run_week7.py train --variant text --seed 42 --gpu 0
# 中断后使用同一配置恢复
python scripts/run_week7.py train --variant text --seed 42 --gpu 0 --resume
~~~

四组 seed 42 在 valid 上筛选后，Text 与候选补齐种子 43/44；冻结方案后再评测 test。完整计划及结果边界见 [实验协议](experiment-protocol.md)。

CPU测试可以验证接口和数学规则；完整训练、解码与指标仍需在实际数据和 GPU 环境中执行。checkpoint 和可重建的大数组不保存在仓库。
