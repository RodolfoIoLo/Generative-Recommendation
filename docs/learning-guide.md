# 🧭 生成式推荐学习路径

本项目把“预测下一部电影”转成“逐位生成电影的语义 ID”。以下用一个虚构样例说明链路；示例中的编号不代表某部真实电影的实验结果。

想查看真实实验中的大张量如何流经 Encoder–Decoder、计算损失并更新参数，见[彩色训练全过程详图](generative-training-diagram.md)（可放大 SVG / 高清 PNG，基于 Week 5–6 冻结配置）。

## 从电影到语义 ID

1. **文本编码**：将电影标题和类型输入 MiniLM，得到 384 维文本向量。
2. **协同表示**：仅用训练集交互构造用户—电影矩阵，通过 SVD 得到每部电影的 64 维协同向量。
3. **信息融合**：文本向量与协同向量分别归一化、加权拼接，再整体归一化，得到 448 维输入。四组对照见 [Week 7 结果](../results/week7-improvement/README.md)。
4. **残差量化**：RQ-VAE 将输入压缩到 32 维潜空间，依次选择三层码本中最接近当前残差的码字。每层的编号组成三位码。
5. **冲突消解**：前三位相同的电影追加组内序号，形成唯一的四位语义 ID。

RQ-VAE 先训练连续自编码器，再用逐层 K-Means 初始化码本，最后联合优化重建和量化损失。它学习重建电影表示；生成模型的推荐损失不会反向更新量化器。

## 从 ID 到 Token

每一位使用独立的 256 个 Token，另有 PAD、BOS 和 EOS 三个特殊 Token：

~~~text
token(position, code) = 3 + position × 256 + code
PAD = 0，BOS = 1，EOS = 2
词表大小 = 4 × 256 + 3 = 1027
~~~

例如，虚构 ID `(37, 128, 5, 0)` 转成 `[40, 387, 520, 771]`。相同数字出现在不同位置会映射到不同 Token。

## 从用户历史到推荐

用户最近最多 20 部电影的 Token 被依次拼接，并以 EOS 结尾，作为 Encoder 输入。训练目标是下一部电影的四个 Token 和 EOS。

~~~text
历史电影 A、B 的 Token → Encoder
目标电影 C 的前面几位  → Decoder
Decoder 预测 C 的下一位 → 交叉熵训练
~~~

训练采用 teacher forcing：预测下一位时提供目标的真实前缀；推理时模型只能使用自己已经生成的前缀。T5 中的 Token embedding 与 RQ-VAE 的 32 维码字是不同的参数。

推理使用 Beam Search 保留候选前缀，并通过动态 Trie 限制：候选必须对应目录中真实存在、且用户完整历史中未出现过的电影。生成的四位 ID 再映射回电影，得到 Top-10。

## 如何判断效果

测试采用 leave-one-out：每名用户有一个保留的目标电影，按全目录排序并屏蔽完整历史，计算 Recall@5/10 和 NDCG@5/10。

旧基线和生成模型结果见 [主实验](../results/week5-6-generative/06-reports/main_results.md)。Week 7 已完成受控三种子训练，Fixed在valid上优于Text；正式test仍需按冻结协议完成，见 [当前结果](../results/week7-improvement/README.md)。
