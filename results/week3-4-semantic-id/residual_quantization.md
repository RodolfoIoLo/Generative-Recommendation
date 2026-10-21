# 阶段 3：残差量化结果

> 数据：MovieLens 1M 的 3,416 个物品；输入为 `all-MiniLM-L6-v2` 生成的
> 384 维单位向量；`L=3`、`K=256`、seed 42。

## 结果

| 方法 | 重建 R² | 三层利用率 | 三层困惑度 | 消解前唯一 ID | 消解前冲突率 | 最大冲突组 | 消解后唯一 ID |
|---|---:|---|---|---:|---:|---:|---:|
| RQ-KMeans | 0.595971¹ | 256 / 256 / 256 | 201.0 / 97.4 / 14.9 | 3,321 | 2.78% | 3 | 3,416 |
| RQ-VAE | 0.623914 | 256 / 253 / 253 | 219.3 / 232.0 / 241.1 | 3,411 | 0.15% | 2 | 3,416 |

均值预测器的 mean-MSE 为 `0.001865811879`、R² 按定义为 0。RQ-VAE 最佳
checkpoint 的 mean-MSE 为 `0.00070171`，明显优于均值预测器。两种方法都在三位
残差量化码之后追加组内序号作为第 4 位；追加后没有 ID 冲突。

¹ 原 RQ-KMeans 日志未输出 R²；这里以三层聚类中心之和作为重建向量，从正式码本
只读复算得到 mean-MSE `0.00075384`、R² `0.595971`。该补充值未参与原实验的模型选择。

## RQ-VAE 训练流程

最初的随机小范围码本初始化发生完全坍塌：三层利用率均为 `1/256`、唯一 ID 为 1、
最佳 R² 为 `-0.0016`。修复后的流程为：

1. 用连续自编码器把 384 维输入压缩到 32 维；训练 200 epochs，最佳 R² 为 `0.6707`。
2. 对全部 3,416 个潜向量逐层执行残差 K-Means，以质心初始化三层 32 维码本。
3. 统一使用“向量内平方和、batch 间平均”的损失口径，以 Adam、学习率 `1e-4`
   联合训练 2,000 epochs。
4. 只在三层码本利用率均不低于 90% 的 checkpoint 中按重建 R² 选优。
5. 追加第 4 位碰撞消歧码，并重新加载 checkpoint 独立验证输出。

正式命令：

```bash
python scripts/build_semantic_id.py \
  --config config/rqvae_ml1m.yaml \
  --emb item_emb.npy --method rqvae --device cuda:0 \
  --checkpoint saved/rqvae_ml1m.pt \
  --out semantic_ids_rqvae.json \
  2>&1 | tee results/rqvae_train.log
```

训练耗时 `191.73 s`。重新加载 checkpoint 后得到 `R²=0.623914`，生成的三层码与
保存的 JSON 完全一致，所有权重均为有限值。

## 文件与校验

| 文件 | 保存位置 | SHA-256 |
|---|---|---|
| RQ-KMeans ID | `semantic_ids_kmeans.json` | `862ebdc71ea6943c53ee34da6a25ff755caaf5b9d35f7e0326905c12a2961183` |
| RQ-VAE ID | `semantic_ids_rqvae.json` | `d3b92f8fc17accf3fcbabf30536b836f22b1da1b6f0aea40004abd758438718a` |
| RQ-KMeans 码本 | `saved/rq_kmeans_ml1m.npz` | `b22f8b4d4766e04c5431858d7f8b46e19a27acaa9193b691a207053bdb6e78c5` |
| RQ-VAE checkpoint | `saved/rqvae_ml1m.pt` | `95a7b03b474958949c466b14c2bbe4697d32263fbb6f393604c62bd8a92f562f` |
| RQ-KMeans 日志 | `results/rq_kmeans.log` | `28397019439576c67e2a902e5f6a1ebce5c1d59eca64b43c02971894e50339d9` |
| RQ-VAE 日志 | `results/rqvae_train.log` | `edee53a5d8c9879eabd446fef11f83277d5e082eeb7a0baad30e9b2364b86f9d` |

仓库保留两份语义 ID、两份干净日志和本报告；二进制码本与 checkpoint 保留在服务器
`~/genrec/saved/`。
