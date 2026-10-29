# 🚀 复现指南

从仓库根目录运行。基础测试只用合成小数据，不需要下载数据集、编码器或模型权重。

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

启动器连续两次确认无计算进程且显存占用不超过 1 GiB 后启动任务。利用率只作提示，避免上一任务结束后的采样残留造成误判；无计算进程时最多复查约 5 秒，仍未空闲则退出。

~~~bash
python scripts/run_week7.py train --variant text --seed 42 --gpu 0
# 中断后使用同一配置恢复
python scripts/run_week7.py train --variant text --seed 42 --gpu 0 --resume
~~~

四组 seed 42 在 valid 上筛选后，Text 与候选补齐种子 43/44；冻结方案后再评测 test。完整计划及结果边界见 [实验协议](experiment-protocol.md)。

CPU测试可以验证接口和数学规则；完整训练、解码与指标仍需在实际数据和 GPU 环境中执行。checkpoint 和可重建的大数组不保存在仓库。
