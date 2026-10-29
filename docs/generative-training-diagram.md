# 生成式推荐模型：训练全过程详图

这张彩色流程图以本仓库 **Week 5–6 冻结实验**为对象，重点展开 Encoder–Decoder 内部发生的计算。主线使用完整 batch 的张量，具体用户样本作为局部放大说明。

[打开可无限放大的 SVG](assets/generative-training-detailed.svg) · [下载高清 PNG（4500 × 12450）](assets/generative-training-detailed.png) · [核验数据与源文件 SHA-256](assets/generative-training-detailed.evidence.json)

[![生成式推荐训练详图预览](assets/generative-training-detailed-preview.png)](assets/generative-training-detailed.svg)

## 推荐阅读顺序

先读第 2、3 区，区分历史 token、监督标签、右移后的 Decoder 输入和共享 embedding；随后沿第 4 区的两列往下读。Encoder 形成整段历史记忆 H，H 在每一层 Decoder 的交叉注意力中提供 K、V。Decoder 的另一路输入来自正确目标的前缀，因果掩码防止当前预测位置看到自己的答案。

再读第 5、6 区：Decoder 的 256 维隐藏状态变成 1,027 个词表分数，交叉熵利用正确 token 的概率计算损失，梯度经过整个模型，累积四个小批量后更新一次参数。第 7 区说明验证时如何真正自回归生成，以及两类训练日志的来源。

## 数字与形状的含义

图中 `B=64、T=81` 表示：使用正式配置中的一个完整小批量，并取该批最长历史达到 20 个物品时的补齐长度。`81 = 20×4+EOS`，目标长度为 `5 = 4 个物品 token + EOS`。这是**由真实配置推导的计算形状**，并非某次训练中已经捕获的激活快照。

代码采用动态补齐，因此某些 batch 的 T 会小于 81；一轮训练最后一批 B 为 51。一个 batch 中的 64 个样本是 64 道前缀预测题，不保证来自 64 个不同用户。验证 batch 为 16，最后一批为 8。

| 计算对象 | 图中常规完整批形状 | 元素数 |
|---|---|---:|
| 历史 token ID / attention mask | `[64,81]` | 各 5,184 |
| Encoder 输入与最终 H | `[64,81,256]` | 各 1,327,104 |
| Encoder 各个 Q/K/V（拆头后） | `[64,8,81,32]` | 各 1,327,104 |
| Encoder 自注意力分数 | `[64,8,81,81]` | 3,359,232 |
| Decoder 输入与各层输出 | `[64,5,256]` | 各 81,920 |
| Decoder 各个 Q/K/V（自注意力拆头后） | `[64,8,5,32]` | 各 81,920 |
| Decoder 因果自注意力分数 | `[64,8,5,5]` | 12,800 |
| Decoder 交叉注意力分数 | `[64,8,5,81]` | 207,360 |
| Encoder 前馈中间特征 | `[64,81,1024]` | 5,308,416 |
| Decoder 前馈中间特征 | `[64,5,1024]` | 327,680 |
| LM Head 输出 logits | `[64,5,1027]` | 328,640 |
| labels | `[64,5]` | 320 |
| 原始训练 loss | 标量 | 1 |

注意力矩阵展示的是数学计算对应的形状；具体后端可能融合计算，并不意味着这些中间矩阵都以独立张量同时驻留显存。BF16 autocast 也不表示所有权重、梯度和激活都具有相同的数据类型。

## 图中真实样本

来源是冻结接口 `train.semantic.jsonl` 的第一行，`user=0`。训练序列有 51 个物品，最后一个训练目标对应重映射物品 ID `1314`。

```text
目标 Semantic ID        [225,160,83,0]
labels                  [228,419,598,771,2]
decoder_input_ids       [1,228,419,598,771]
Encoder 历史             第31～50个物品，共20个；之后附加EOS
```

对应的五个训练条件为：历史+BOS 预测 228；历史+BOS+228 预测 419；依此类推，最后预测 EOS。模型通过掩码在一次训练前向中并行处理五个位置。这里的 771 是第四位消歧码 0 对应的 token ID，和 PAD=0 不同。

图中没有提供未经记录的隐藏向量分量、注意力热度、逐 token 概率或单样本损失。日志中的 loss 是最近 100 个 optimizer steps 内的小批量原始损失均值；不能将其解释为放大样本的损失。图中 seed42 指标也不是三随机种子的平均成绩。

## 参数账本与来源

| 参数部分 | 数量 |
|---|---:|
| 共享 embedding `1027×256` | 262,912 |
| 6 层 Encoder、位置偏置及末尾归一化 | 4,722,176 |
| 6 层 Decoder、位置偏置及末尾归一化 | 6,296,576 |
| 总计 | **11,281,664** |

LM Head 与共享 embedding 绑定，不另算一份输出矩阵。物品文本编码器、RQ-VAE 和已经导出的离散编码不在本阶段反向传播的计算图中。

仓库内证据：

- [正式配置](../config/generative_ml1m.yaml)：实际模型维度、训练与解码设置。
- [数据集、词表、collator 与评估实现](../scripts/generative_common.py)：前缀展开、历史截断、补齐、token 偏移、Trie 和指标。
- [训练实现](../scripts/train_generative.py)：初始化、优化器、自动混合精度、损失、梯度累积、早停与 checkpoint。
- [审计日志](../results/week5-6-generative/01-pipeline-validation/audit.log)：真实训练规模与参数量。
- [冻结语义接口](../results/week3-4-semantic-id/05-frozen-interface/)：真实用户序列、Semantic ID、词表与元数据。
- [seed42 训练历史](../results/week5-6-generative/03-formal-training/t5_sid_seed42_history.jsonl)：loss 和 valid 指标。

T5 的归一化、子层顺序、共享权重、输出缩放和标签右移同时参考 [Hugging Face 官方 T5 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py)及[官方配置](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/configuration_t5.py)。上游链接指向 main，可能继续变化；本仓库实验依赖版本记录在 [server-requirements.txt](../environment/server-requirements.txt)，实验参数与统计以本地代码、配置和日志为准。

## 重建与文档引用

构建脚本会核验配置中的关键维度、数据统计、样本和参数量，再输出 SVG 及证据 JSON。SVG 生成只需要 Python 标准库；可选 Pillow 用于更准确地测量文字。PNG 导出需要 Node.js 的 sharp。

```bash
# 生成 SVG 与 evidence.json
python scripts/render_training_diagram.py

# 安装/提供 sharp 后，同时导出高清 PNG 与预览
python scripts/render_training_diagram.py --sharp-module sharp
```

也可以用 `--node` 指定 Node 可执行文件，或用 `--sharp-module` 指定 sharp 模块的绝对路径。默认 `--scale 1.5`，输出 PNG 为 4500×12450。文字采用 Microsoft YaHei / SimHei 中文字体；其他系统应安装相应字体或在脚本中调整字体设置后重建。

在仓库根目录的 Markdown 中插图：

```markdown
[![生成式推荐模型训练全过程](docs/assets/generative-training-detailed-preview.png)](docs/assets/generative-training-detailed.svg)
```

图片覆盖本仓库训练算法的数据流与控制流，包括一次前向、一次反向、四批累积、优化器更新、验证与保存；没有展开矩阵乘法的 CUDA 内核、显存分配和文件系统底层操作。
