# 🎞️ 文本与观看行为的融合输入

| 文件 | 用途 |
|---|---|
| [meta.json](meta.json) | train-only来源、SVD设置、数组形状和原始SHA-256 |
| [weights.json](weights.json) | 每部电影在Text／Fixed／Fixed-Match／Adaptive中的协同权重 |
| [neighbors.json](neighbors.json) | 固定20部电影在协同空间中的近邻与余弦相似度 |

文本384维，协同64维；两者分别归一化后加权拼接，再整体归一化。Text的协同部分置0，四组输入统一为`3416×448`。协同向量来自train的二值用户—物品矩阵，权重频次则统计train交互次数；均不读取valid/test。数组缓存放在实验工作区`work/week7/`。
