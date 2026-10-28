# Week 7：文本—协同融合式语义标识符

已完成输入冻结、CPU冒烟及恢复、协同特征构建，以及四组RQ-VAE量化与语义ID导出。生成模型的融合消融和正式评测尚未开展，本目录暂不包含新的推荐效果结论。

## 目录与证据

| 路径 | 内容 |
|---|---|
| [01-diagnostics/environment.json](01-diagnostics/environment.json) | 服务器Python与依赖版本 |
| [01-diagnostics/input_sha256.txt](01-diagnostics/input_sha256.txt) | Step 7.0冻结输入哈希 |
| [01-diagnostics/reused_code_sha256.txt](01-diagnostics/reused_code_sha256.txt) | 复用旧脚本的哈希 |
| [01-diagnostics/cpu-smoke/](01-diagnostics/cpu-smoke/) | 两份CPU冒烟源记录，内容保持服务器原样 |
| [02-embeddings/meta.json](02-embeddings/meta.json) | SVD参数、输入/数组哈希、维度与权重定义 |
| [02-embeddings/weights.json](02-embeddings/weights.json) | 四组逐物品协同权重，第i项对应item i |
| [02-embeddings/neighbors.json](02-embeddings/neighbors.json) | 20个固定锚点的Top-10协同近邻 |
| [03-semantic-ids/](03-semantic-ids/README.md) | 四组ID、码本质量、训练日志与前缀分析 |

CPU冒烟源文件位于服务器results/week7-improvement/05-formal-training/；仓库将其归入01-diagnostics/cpu-smoke/，避免与正式训练混淆。两份文件名称和内容均保持不变，仅调整展示目录。

## 已验证结果

- 6,040名用户、3,416部电影、987,531次训练交互，对应981,491个全前缀训练样本。
- 11项CPU单元测试全部通过，无失败、错误或跳过。
- 小模型从step2恢复到step4；在step2和4分别对8名用户进行约束验证，非法ID、已见物品和短列表率均为0。
- 所有新向量均为float32、有限值、单位范数；四组融合输入均为3416×448，逐行范数最大误差小于1.8e-7。
- 保存的频次与train重算一致；四组权重与公式一致；融合数组与重算结果一致；哈希与metadata一致。
- 四组checkpoint均可重现全部保存的ID；train、valid、test各6,040行，经映射后与原始物品和目标逐条一致。

冒烟Recall/NDCG为0，只验证前向、反向、保存、恢复和解码。best保留在step2，last在step4；smoke的finished=false表示未跑满其20步日程，不代表4步恢复检查失败。

## 协同向量如何得到

仅用train构造6040×3416用户—物品二值矩阵X，对X的转置做TruncatedSVD，得到3416×64物品表示并逐行归一化。相近向量表示共同交互结构相近；不使用文本或valid/test交互，也不保留观看顺序。

参数n_components=64、n_iter=10、seed=42。解释方差比之和为0.439468，约43.95%；这是压缩诊断值，不是Recall、准确率或推荐收益。

## 四组融合输入

文本向量t与协同向量c分别归一化后，输入为normalize([t; λc])。整体归一化后，协同部分平方范数占比为λ²/(1+λ²)，与维度数384:64不等价。

| 组别 | 协同权重λ | 协同平方范数占比 | 对照目的 |
|---|---:|---:|---|
| text | 0 | 0% | 同448维结构的纯文本控制 |
| fixed | 1 | 50% | 固定强度融合 |
| fixed_match | 0.765089 | 约36.92% | 匹配adaptive的整体协同强度 |
| adaptive | 0.262873–1 | 单物品约6.46%–50%，均值约36.92% | 按物品训练频次分配 |

adaptive采用clip(log(1+freq)/log(1+q80),0,1)。训练频次q80=455，684部电影达到权重1；训练频次最小4、中位数144、最大3377。完整数据执行5-core后再切分，故train最小频次可小于5。

fixed_match与adaptive整体协同占比相同，分配方式不同。它检验收益是否来自逐物品权重，而不只是平均协同强度变化。占比是表示空间中的数值量，不是最终推荐贡献比例。

## 协同近邻示例

以下仅由train交互与SVD得到，电影名称用于阅读，不参与SVD：

| 锚点 | 近邻 | 协同余弦 |
|---|---|---:|
| Toy Story (1995) | Toy Story 2 (1999) | 0.9132 |
| Toy Story (1995) | Babe (1995) | 0.8750 |
| Toy Story (1995) | Aladdin (1992) | 0.8618 |
| Species (1995) | Mimic (1997) | 0.9238 |
| Species (1995) | Alien: Resurrection (1997) | 0.9145 |
| Species (1995) | Species II (1998) | 0.9014 |

这些例子表明表示包含可读的共同交互结构，尚不能证明融合ID会提高推荐指标。

## 服务器保留

~~~text
~/genrec/work/week7/
├── collab.npy             # 3416×64
├── text.npy               # 3416×448
├── fixed.npy              # 3416×448
├── fixed_match.npy        # 3416×448
├── adaptive.npy           # 3416×448
└── frequency.json
~~~

数组和checkpoint不进入Git；四组语义ID映射、代码、配置及轻量质量证据保留在仓库。下一步可用空闲GPU进行搜索诊断和生成模型训练。
