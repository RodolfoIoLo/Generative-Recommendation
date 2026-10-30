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

## 4. 后续训练与评测

以下为训练入口示例，卡号按实际空闲设备指定。对应变体的量化接口须先完成：

启动器连续两次确认无计算进程且显存占用不超过 1 GiB 后启动任务。利用率只作提示，避免上一任务结束后的采样残留造成误判；无计算进程时最多复查约 5 秒，仍未空闲则退出。

~~~bash
python scripts/run_week7.py train --variant text --seed 42 --gpu 0
# 中断后使用同一配置恢复
python scripts/run_week7.py train --variant text --seed 42 --gpu 0 --resume
~~~

恢复会沿用保存的优化器、学习率、随机数和数据位置。checkpoint先写入临时目录，完整写入后再替换旧版本；存储空间不足512 MiB或写入时磁盘满，会等待并重试。恢复时会核对最佳权重与原始valid记录并补齐缺失元数据；当last正好对应最佳验证步时，也可据此重建best。无法找回更早的最佳权重时会明确报错。

四组 seed 42 在 valid 上筛选后，Text 与候选补齐种子 43/44；冻结方案后再评测 test。完整计划及结果边界见 [实验协议](experiment-protocol.md)。

已完成的筛选运行、h20基线与候选记录见 [训练记录](../results/week7-improvement/05-formal-training/README.md)和 [验证集筛选](../results/week7-improvement/04-screening/README.md)。可用 `python scripts/run_week7.py curves` 从完成的正式训练记录重建曲线；该命令兼容服务器旧平铺目录与仓库分类目录，并排除冒烟及进行中的第七周运行。

CPU测试可以验证接口和数学规则；完整训练、解码与指标仍需在实际数据和 GPU 环境中执行。checkpoint 和可重建的大数组不保存在仓库。

## 5. 正式评测与本地打包

四组seed42完成后，只在valid上选择一次候选：

~~~bash
python scripts/run_week7.py select
~~~

随后Text与selected补齐seed43/44，并训练两种h20基线；以下为单次入口，基线种子需覆盖42、43、44：

~~~bash
python scripts/run_week7.py train --variant text --seed 43 --gpu 0
python scripts/run_week7.py train --variant selected --seed 43 --gpu 0
python scripts/run_week7.py baseline --model SASRec --seed 42 --split train --data_path "$GENREC_DATA_ROOT/recbole" --gpu 0
python scripts/run_week7.py baseline --model GRU4Rec --seed 42 --split train --data_path "$GENREC_DATA_ROOT/recbole" --gpu 0
~~~

生成模型补充种子还需执行seed44。六个生成模型齐备后，用可重复执行的入口补齐全部12项test：

~~~bash
python scripts/week7_test_queue.py --status
python scripts/week7_test_queue.py --gpu 0 --data_path "$GENREC_DATA_ROOT/recbole" --wait
python scripts/run_week7.py summarize
~~~

续跑入口核对模型指纹、完整预测和样例后跳过已完成项，只补缺失或不完整的结果。`--wait`表示卡被占用时等待；不加则在卡忙时退出。通过tmux运行可在断开SSH后继续，Ctrl+C停止后可用同一命令恢复。

完整预测保存在 `work/week7/predictions/`，供指标复核与bootstrap使用；仓库只收录指标和固定20名用户样例。

发布时先将已验收结果同步回公开仓库，再从仓库根目录执行：

~~~bash
python scripts/week7_release.py --check-only
python scripts/week7_release.py
~~~

打包会验证完整结果、生成 `dist/week7-public.zip` 并逐文件校验内容，不重写README、许可证或人工文档，也不生成同步清单。
