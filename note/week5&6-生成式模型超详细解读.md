# 第五至六周：生成式推荐模型超详细解读

> 本文面向第一次接触深度学习与生成模型的读者，按数据真正经过程序的顺序，解释 MovieLens 1M 上的 T5 式生成推荐模型。每一节固定回答三件事：最小例子中发生了什么、真实训练时张量扩大成什么规模、仓库中的哪段代码完成了这件事。
>
> 文中的“教学代码”只保留关键运算；标为“项目代码”的片段来自当前仓库。完整实现见 [`generative_common.py`](../scripts/generative_common.py)、[`train_generative.py`](../scripts/train_generative.py) 与 [`evaluate_generative.py`](../scripts/evaluate_generative.py)。

---

## 0. 先看全程：模型究竟在学习什么

一条训练题可以写成：

```text
历史：电影 A → 电影 B → 电影 C
答案：电影 D
```

电影 D 不再用单个物品编号表示，而是用第三至四周冻结的四位语义 ID 表示。例如：

```text
电影 D → [22, 235, 4, 0]
```

前 3 位由 RQ-VAE 产生，第 4 位是冲突消歧码。生成模型要完成的任务是：

```text
读懂 A、B、C 的语义 ID
  → 依次预测 D 的第 1、2、3、4 位
  → 预测 EOS，表示一个物品生成完毕
```

整个训练流程如下：

```text
冻结的用户序列与语义 ID
        ↓
构造“历史 → 下一物品”训练样本
        ↓
组成 batch，并补 PAD
        ↓
Embedding：token 整数 → 可学习向量
        ↓
Encoder：让每个历史位置读取整段历史
        ↓
Decoder：结合历史和正确前缀，预测 5 个位置
        ↓
输出层：每个位置产生 1,027 个 token 分数
        ↓
交叉熵：比较预测与正确 token
        ↓
反向传播：计算每个参数应怎样调整
        ↓
AdamW：更新模型参数
        ↓
验证：Beam Search + Trie 生成 Top-10 物品
```

这里最重要的分界是：

- **前向传播**使用当前参数算出预测和损失；
- **反向传播**计算损失对各参数的梯度；
- **优化器 step**才真正修改参数；
- **验证与推荐**只使用已经学好的参数，不做反向传播。

---

## 1. 冻结接口：四位语义 ID 如何变成模型 token

### 1.1 最小示例

假设一个物品的四位语义 ID 是：

```text
[37, 128, 5, 0]
```

四个位置上的数字不能直接共用 token。第一位的 `37` 表示第一层码本中的 37，第二位的 `37` 表示第二层码本中的 37，两者含义不同。

项目给不同位置划分独立的 token 区间：

| 内容 | token 范围 |
|---|---:|
| PAD / BOS / EOS | 0 / 1 / 2 |
| 第 1 位语义码 | 3～258 |
| 第 2 位语义码 | 259～514 |
| 第 3 位语义码 | 515～770 |
| 第 4 位消歧码 | 771～1026 |

第 `level` 位的映射公式是：

```text
token = 3 + level × 256 + code
```

因此：

```text
[37, 128, 5, 0]
       ↓
[40, 387, 520, 771]
```

### 1.2 真实数据规模

当前冻结接口包含 3,416 个物品，每个物品都有唯一四位 ID。总词表为：

```text
3 个特殊 token + 4 × 256 个位置 token = 1,027
```

真实物品示例：

| 电影 | 四位语义 ID | 送入模型的四个 token |
|---|---|---|
| `Toy Story (1995)` | `[204,158,180,0]` | `[207,417,695,771]` |
| `Mulan (1998)` | `[22,235,4,0]` | `[25,494,519,771]` |
| `Pocahontas (1995)` | `[225,40,11,0]` | `[228,299,526,771]` |

这些数字来自冻结后的 `semantic_ids.json`，不是人为编写的类别标签。单个相似电影对不一定共享前缀；“共享前缀的物品整体上更相似”是通过全体物品统计验证的群体规律。

### 1.3 对应项目代码

[`SemanticArtifacts.code_to_tokens()`](../scripts/generative_common.py) 中的真实映射：

```python
def code_to_tokens(self, code):
    if len(code) != self.length or any(not 0 <= int(x) < self.k for x in code):
        raise ValueError(f"bad semantic code: {code}")
    return [3 + level * self.k + int(value)
            for level, value in enumerate(code)]
```

这段代码只做确定性编号转换，不含神经网络，也不产生新的语义。语义已经由 RQ-VAE 和消歧步骤冻结。

---

## 2. 一条用户序列如何展开成许多训练题

### 2.1 最小示例

假设某用户训练序列为：

```text
A → B → C → D
```

它会产生三条训练样本：

```text
历史 [A]       → 答案 B
历史 [A,B]     → 答案 C
历史 [A,B,C]   → 答案 D
```

长度为 `n` 的序列产生 `n-1` 条样本。第一部电影之前没有历史，所以不能形成训练题。

假设 D 的语义 ID 是 `[22,235,4,0]`，该样本的标签为：

```text
[25, 494, 519, 771, EOS]
```

历史中的每部电影同样转换成 4 个 token，所有历史 token 展平后，末尾追加一个 EOS。

### 2.2 真实数据规模

当前训练集有：

```text
987,531 次训练交互
6,040 名用户
```

因此训练题数量是：

```text
987,531 - 6,040 = 981,491
```

模型最多读取最近 20 部电影，因此 Encoder 最大输入长度为：

```text
20 部 × 每部 4 token + 1 个 EOS = 81 token
```

目标始终是：

```text
4 个物品 token + 1 个 EOS = 5 token
```

### 2.3 对应项目代码

[`TrainPrefixDataset.__getitem__()`](../scripts/generative_common.py) 中的真实逻辑：

```python
items = self.rows[row_idx]["items"]
source = self.a.encode_items(
    items[:target_pos],
    self.max_history_items,
)
labels = self.a.code_to_tokens(items[target_pos]) + [self.a.eos]

return {"input_ids": source, "labels": labels}
```

`items[:target_pos]` 是历史，`items[target_pos]` 是下一物品。数据集没有提前复制近百万条长序列，而是用累计样本数和二分查找，在访问某个索引时即时定位用户与目标位置。

---

## 3. 多条训练题如何组成 batch

### 3.1 最小示例

假设一个 batch 有三条历史，其 token 长度分别为 5、9、13：

```text
样本 1：[真实 token × 5]
样本 2：[真实 token × 9]
样本 3：[真实 token × 13]
```

GPU 适合处理规则矩形，所以当前 batch 补到最长的 13：

```text
input_ids：
[真实 × 5 ][PAD × 8]
[真实 × 9 ][PAD × 4]
[真实 × 13]

attention_mask：
[1 × 5 ][0 × 8]
[1 × 9 ][0 × 4]
[1 × 13]
```

PAD 只负责占位；mask 告诉注意力哪些位置没有真实内容。不同样本之间不会互相读取，它们只是共享同一套模型参数并行计算。

### 3.2 真实张量规模

实际 `batch_size=64`。如果当前 batch 的最长历史为 `L`：

```text
input_ids       [64, L]       L ≤ 81
attention_mask  [64, L]
labels          [64, 5]
```

程序按“当前 batch 的最长序列”补齐，并非每个 batch 都补到 81。这样可以减少无效计算。

### 3.3 对应项目代码

[`TrainCollator`](../scripts/generative_common.py) 中的真实代码：

```python
max_x = max(len(x["input_ids"]) for x in batch)
max_y = max(len(x["labels"]) for x in batch)

ids.append(x + [self.pad] * (max_x - len(x)))
masks.append([1] * len(x) + [0] * (max_x - len(x)))
labels.append(y + [-100] * (max_y - len(y)))
```

输入端 PAD 用 `attention_mask=0` 屏蔽；标签端若需要补齐，则填 `-100`，交叉熵会忽略这些位置。当前目标通常固定为 5 个 token，但保留通用补齐逻辑更稳妥。

---

## 4. “神经网络”到底是什么：先看一个只有两个参数的模型

### 4.1 最小示例

一个最小模型可以只有两次乘法和一次加法：

```python
x = [2.0, 1.0]
w = [0.5, -0.2]       # 可训练参数

output = 2.0 * 0.5 + 1.0 * (-0.2)
# output = 0.8
```

如果参数改成 `[0.8,0.1]`，相同输入会得到 `1.7`。网络的结构规定“怎样计算”，参数规定“这次具体算成什么”。

希望同时得到 256 个输出，就安排 256 套这样的加权求和，并把参数排成矩阵：

```python
linear = torch.nn.Linear(input_dim, 256)
output = linear(x)
```

所谓“层”，就是一组可重复使用、包含可训练参数的计算。所谓“深”，就是上一层输出继续送给下一层。

### 4.2 真实模型规模

本项目模型由 [`build_model()`](../scripts/train_generative.py) 随机初始化：

```python
model_cfg = T5Config(
    vocab_size=1027,
    d_model=256,
    d_ff=1024,
    num_layers=6,
    num_decoder_layers=6,
    num_heads=8,
    d_kv=32,
    dropout_rate=0.1,
    feed_forward_proj="relu",
)

model = T5ForConditionalGeneration(model_cfg)
```

它有 11,281,664 个参数。这些参数在训练开始时随机初始化；这里没有加载自然语言 T5 的预训练 checkpoint。

384 维 `item_emb.npy` 也不会直接送入这个 T5。它已经在上一阶段帮助产生四位语义 ID，当前模型只读取离散 token。

---

## 5. Embedding：整数编号如何变成可计算的向量

### 5.1 最小示例

假设词表只有 4 个 token，每个 token 用 3 个数字表示：

```python
embedding = torch.nn.Embedding(4, 3)
```

内部是一张 `4×3` 的参数表：

```text
token 0 → [ 0.1,  0.3, -0.2]
token 1 → [-0.4,  0.2,  0.7]
token 2 → [ 0.8, -0.1,  0.5]
token 3 → [ 0.0,  0.6, -0.3]
```

输入 `[2,1]` 就是取第 2 行和第 1 行：

```text
[[ 0.8, -0.1, 0.5],
 [-0.4,  0.2, 0.7]]
```

token 编号本身没有大小关系；有意义的是训练后对应的向量。

### 5.2 真实张量规模

真实 embedding 表形状是：

```text
[1027, 256]
```

一个输入 batch 经过查表：

```text
input_ids  [64, L]
    ↓
embedding  [64, L, 256]
```

T5 让 Encoder、Decoder 共用 token embedding，并把它与最终词表输出层绑定。一个 token 的表示既用于“读取”，也参与“预测这个 token 应得多少分”。

Embedding 参数会接收梯度。训练可以逐渐调整：哪些 token 表示容易被注意力匹配，哪些组合有助于预测下一物品。

---

## 6. Encoder 自注意力：Q、K、V 和“头”究竟是什么

这一节从零解释自注意力。先给出结论：**自注意力不是在一个现成的“知识库”里查资料，而是模型根据当前 batch 的输入，临时计算“每个位置应该从其他位置读取多少信息”。**

### 6.1 为什么仅有 Embedding 还不够

Embedding 查表之后，每个 token 都得到一个 256 维向量。但此时不同位置彼此孤立：

```text
位置 1 的向量只知道“自己是什么 token”
位置 2 的向量只知道“自己是什么 token”
位置 3 的向量只知道“自己是什么 token”
```

假设同一个语义 token 分别出现在两段历史里：

```text
历史甲：Toy Story → A Bug's Life → Antz
历史乙：Saving Private Ryan → Platoon → Antz
```

查 Embedding 表时，两处 `Antz` 的初始向量完全相同。但它出现在动画片历史和战争片历史中时，对下一步推荐的作用可能不同。Encoder 因而要把：

```text
“这个位置本身是什么”
```

加工为：

```text
“这个位置处在当前整段历史中时，应该表示什么”
```

自注意力负责让各位置交换信息。注意，模型实际处理的是展平后的**语义 token 位置**，一部电影占 4 个位置；它不会天然知道哪四个位置属于一部电影，只能通过固定顺序、相对位置和大量训练样本学习这种结构。

### 6.2 Q、K、V 的直观含义

Q、K、V 分别是 Query、Key、Value：

| 名称 | 可以先理解为 | 回答的问题 |
|---|---|---|
| Query（查询） | 当前位置写出的“检索需求” | 我现在想从其他位置找什么？ |
| Key（键） | 每个被检索位置提供的“匹配标签” | 我的内容是否符合这个查询？ |
| Value（值） | 每个位置真正准备交付的内容 | 如果选中我，应该把什么信息传过去？ |

可以把它类比成一次检索：

```text
Query 和 Key 比较       → 决定匹配程度
匹配程度经过 Softmax    → 决定每个位置的读取比例
按比例混合各位置 Value  → 得到真正读取到的信息
```

为什么要拆成三个东西，而不是直接拿原向量互相比较？因为“用什么特征判断相关”和“相关后传递什么内容”不一定相同。

例如，在推荐中可能出现这样的需求：

- 匹配时更关心电影类型与时间远近；
- 真正传递时还需要携带更丰富的语义与行为上下文；
- 当前正在生成第一位粗粒度语义码和第四位消歧码时，所需信息也可能不同。

Q/K/V 让模型能够分别学习“如何发出需求”“如何被匹配”和“如何提供内容”。这些含义不是人工写死的，只是它们在计算中的角色。

### 6.3 Q、K、V 到底来自哪里

设某一层收到的位置向量为 `X`。每个位置都是 256 维，整个 batch 的形状为：

```text
X：[batch_size, 序列长度, 256]
```

这一层内部保存三组可训练矩阵：

```text
W_Q：把输入向量转换成 Query
W_K：把输入向量转换成 Key
W_V：把输入向量转换成 Value
```

计算为：

```text
Q = X × W_Q
K = X × W_K
V = X × W_V
```

本文为了直观，把矩阵写成“输入乘权重”。PyTorch 的 `nn.Linear` 在内存中把权重保存为 `[输出维, 输入维]`，代码内部实际使用权重的转置；这只是存放约定不同，完成的仍是同一种加权求和。

所以，Q、K、V 都来自**同一批输入向量 X 的三种不同变换**，不是数据文件提前提供的，也不是程序员为每部电影手工填写的。

必须区分“模型参数”和“本次计算产生的临时张量”：

| 对象 | 从哪里来 | 是否长期保存 | 谁负责改变它 |
|---|---|---|---|
| `W_Q/W_K/W_V` | 模型初始化时创建 | 是，写入 checkpoint | 反向传播计算梯度，AdamW 更新 |
| `X` | Embedding 或上一层输出 | 仅在本次前向中存在 | 由前面网络根据输入计算 |
| `Q/K/V` | `X` 乘三组参数矩阵 | 仅在本次前向中存在 | 每个 batch 重新计算 |
| 注意力分数与权重 | Q 和 K 计算得到 | 仅在本次前向中存在 | 每个输入都不同 |

因此，“谁在维护 Q、K、V”这个问题需要分两层回答：

- 真正被长期训练和保存的是产生它们的参数矩阵 `W_Q/W_K/W_V`；
- Q、K、V 本身不是长期知识，每次输入新的用户历史都会重新产生。

而且每一层都有自己独立的 Q/K/V 参数。第 1 层的 `W_Q` 和第 6 层的 `W_Q` 不是同一份参数。

### 6.4 用二维数字把来源完整算一遍

为了能手算，暂时假设只有三个位置，每个位置只有 2 维：

```text
位置 A 的输入向量 x_A = [1, 0]
位置 B 的输入向量 x_B = [0, 1]
位置 C 的输入向量 x_C = [1, 1]
```

把三个位置叠成矩阵：

```text
X = [[1, 0],
     [0, 1],
     [1, 1]]
```

为了让例子简单，假设：

```text
W_Q = [[1, 0],       W_K = [[1, 0],
       [0, 1]]              [0, 1]]

W_V = [[1, 1],
       [0, 1]]
```

于是：

```text
Q = X × W_Q = [[1,0], [0,1], [1,1]]
K = X × W_K = [[1,0], [0,1], [1,1]]
V = X × W_V = [[1,1], [0,1], [1,2]]
```

现在只看位置 C。它的 Query 是：

```text
Q_C = [1,1]
```

用 `Q_C` 分别和三个 Key 做点积：

```text
Q_C · K_A = [1,1] · [1,0] = 1
Q_C · K_B = [1,1] · [0,1] = 1
Q_C · K_C = [1,1] · [1,1] = 2
```

分数 `[1,1,2]` 还不是比例。Softmax 把它们转换为和为 1 的正数：

```text
Softmax([1,1,2]) ≈ [0.212, 0.212, 0.576]
```

这表示位置 C 此次读取：

```text
21.2% 的 V_A
21.2% 的 V_B
57.6% 的 V_C
```

最终读到的上下文为：

```text
context_C
= 0.212×[1,1] + 0.212×[0,1] + 0.576×[1,2]
= [0.788, 1.576]
```

注意两点：

1. 用来计算比例的是 `Q_C` 和各个 `K`；
2. 最后被加权混合的是各个 `V`，不是 `K`。

下面是同一过程的最小代码，每行都标出了含义：

```python
import torch

# 三行代表三个序列位置；每行只有两个数字，便于手算。
X = torch.tensor([
    [1.0, 0.0],  # 位置 A 的输入表示
    [0.0, 1.0],  # 位置 B 的输入表示
    [1.0, 1.0],  # 位置 C 的输入表示
])

# 这三张矩阵在真实模型中都是需要训练的参数。
# 这里手工给定数值，只是为了展示一次前向计算。
W_Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
W_K = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
W_V = torch.tensor([[1.0, 1.0], [0.0, 1.0]])

# 同一份输入 X 经过三种线性变换，产生 Q、K、V。
Q = X @ W_Q  # [3,2]：每个位置提出什么查询
K = X @ W_K  # [3,2]：每个位置如何接受匹配
V = X @ W_V  # [3,2]：每个位置真正提供什么内容

# 所有 Query 同时和所有 Key 比较。
# scores[i,j] 表示位置 i 想从位置 j 读取多少。
scores = Q @ K.T  # [3,3]

# 对每一行单独做 Softmax。
# 每个查询位置分配给所有来源位置的权重之和都是 1。
weights = torch.softmax(scores, dim=-1)  # [3,3]

# 用注意力权重混合 Value。
# context[i] 是位置 i 从整个序列读取后的新信息。
context = weights @ V  # [3,2]

print(scores)
print(weights)
print(context)
```

真实训练不会手工指定这些矩阵。它们最初是随机值，之后由“下一物品预测是否更准确”这个最终目标间接训练。

### 6.5 从一个位置扩展到整张注意力分数表

上面的例子只手算了位置 C。实际程序一次计算所有位置：

```text
scores = Q × Kᵀ
```

如果序列长度为 `L`，`scores` 就是 `L×L`：

| | 被读取 A | 被读取 B | 被读取 C |
|---|---:|---:|---:|
| 查询 A | `Q_A·K_A` | `Q_A·K_B` | `Q_A·K_C` |
| 查询 B | `Q_B·K_A` | `Q_B·K_B` | `Q_B·K_C` |
| 查询 C | `Q_C·K_A` | `Q_C·K_B` | `Q_C·K_C` |

每一行代表“一个位置准备如何读取全序列”。Softmax 按行执行，所以每一行的读取比例都独立归一化。

标准 Transformer 经常把点积除以 `sqrt(d_k)`，防止维度变大后分数过大。本项目所用 T5 实现没有这一步显式缩放，即缩放系数为 `1.0`；它会在 Softmax 之前加入**相对位置偏置**和 **mask**：

```text
最终分数 = QKᵀ + 相对位置偏置 + mask
```

- **相对位置偏置**告诉注意力两个 token 大致相隔多远。T5 不使用普通的绝对位置向量，而是学习按相对距离分桶的偏置；
- **PAD mask** 把补齐位置的分数变成极大的负数，Softmax 后接近 0；
- Encoder 没有因果限制，因为输入全是已经发生的历史，任意真实历史位置可以读取其他真实历史位置。

### 6.6 “头”到底是什么

到目前为止，整套 Q/K/V 只会产生**一种读取方案**。但同一段用户历史可能同时存在多种有用关系：

- 某些关系与类型相近有关；
- 某些关系与最近发生的交互有关；
- 某些关系可能连接同一物品内部的四位 token；
- 某些关系可能捕捉重复出现的粗粒度语义码。

如果只有一张注意力权重表，所有需求必须挤在同一种混合方式中。多头注意力的做法是：**并行建立多套较小的 Q/K/V 子空间，让每套独立计算一张注意力表。每一套就叫一个 head。**

本项目设置：

```text
模型维度 d_model = 256
头数 num_heads = 8
每头维度 d_kv = 32
8 × 32 = 256
```

一次真实计算可按下面的顺序理解：

```text
每个位置 256 维
    ↓ 分别乘 W_Q、W_K、W_V
Q/K/V 各得到 256 维
    ↓ 重排形状
拆成 8 份，每份 32 维
    ↓
8 个头各自计算 QKᵀ、Softmax、Value 加权和
    ↓
8 份 32 维结果重新拼成 256 维
    ↓ 乘输出矩阵 W_O
得到该注意力模块的 256 维输出
```

“头”不是：

- 8 个独立模型；
- 8 组用户；
- 8 部电影；
- 人工规定好的“类型头”“时间头”。

它只是同一注意力层里的 8 条并行信息读取通道。程序没有规定每个头必须学习什么。某个头若形成一种读取模式并能持续降低最终交叉熵，反向传播就会强化对应参数；如果没有帮助，也可能学不到容易解释的功能。

代码实现通常不会真的写 8 个 Python 循环，而是用一个大矩阵乘法得到 256 维，再把它 reshape 为 `8×32`。因此“每个头有不同参数”在实现上表现为大参数矩阵中的不同切片。

### 6.7 真实 batch 中每一步的形状

令：

```text
B = 64      batch 中样本数
L ≤ 81      当前 batch 的历史 token 长度
H = 8       头数
Dh = 32     每头维度
D = 256     模型维度，等于 H×Dh
```

形状变化为：

```text
输入 X                    [64, L, 256]
线性变换后的 Q/K/V         [64, L, 256]
拆成 8 个头               [64, 8, L, 32]
Q 与 K 两两比较            [64, 8, L, L]
Softmax 后的权重           [64, 8, L, L]
权重乘 V                  [64, 8, L, 32]
拼回所有头                [64, L, 256]
经过输出投影 W_O           [64, L, 256]
```

下面的代码与真实计算等价，但保留了完整注释。为突出结构，省略了缓存等工程细节：

```python
import torch
import torch.nn.functional as F

B, L = 64, 81       # batch 大小与本批最长序列长度
H, Dh = 8, 32       # 8 个头，每头 32 维
D = H * Dh          # 总模型维度 256

# x 是 Embedding 或上一层输出；这里用随机数代替真实输入。
x = torch.randn(B, L, D)                         # [64,81,256]

# 四个线性层包含真正会训练并写入 checkpoint 的参数。
# T5 的注意力投影不使用 bias。
q_proj = torch.nn.Linear(D, H * Dh, bias=False)
k_proj = torch.nn.Linear(D, H * Dh, bias=False)
v_proj = torch.nn.Linear(D, H * Dh, bias=False)
o_proj = torch.nn.Linear(H * Dh, D, bias=False)

# 对每个位置做三种不同的线性变换。
q = q_proj(x)                                      # [64,81,256]
k = k_proj(x)                                      # [64,81,256]
v = v_proj(x)                                      # [64,81,256]

# 把最后的 256 维拆成 8 个头 × 32 维。
# transpose 后，head 维提前，便于每个头独立做矩阵乘法。
q = q.view(B, L, H, Dh).transpose(1, 2)            # [64,8,81,32]
k = k.view(B, L, H, Dh).transpose(1, 2)            # [64,8,81,32]
v = v.view(B, L, H, Dh).transpose(1, 2)            # [64,8,81,32]

# 每个查询位置与每个来源位置比较。
# 最后两个维度从 [81,32] × [32,81] 变为 [81,81]。
scores = q @ k.transpose(-2, -1)                   # [64,8,81,81]

# 真实 T5 会在这里加相对位置偏置。
# scores = scores + position_bias

# pad_mask 的真实位置为 True，PAD 位置为 False。
pad_mask = torch.ones(B, L, dtype=torch.bool)       # 示例中暂设全为真实 token

# 扩成 [B,1,1,L] 后，可同时广播到所有头和所有查询位置。
# PAD 来源位置被设成极小值，Softmax 后权重约等于 0。
scores = scores.masked_fill(
    ~pad_mask[:, None, None, :],
    torch.finfo(scores.dtype).min,
)

# 对“被读取位置”这一维做 Softmax。
weights = F.softmax(scores, dim=-1)                # [64,8,81,81]

# 每个头使用自己的权重混合自己的 Value。
context = weights @ v                              # [64,8,81,32]

# 把 8 个头重新拼成每个位置 256 维。
context = context.transpose(1, 2).contiguous()     # [64,81,8,32]
context = context.view(B, L, H * Dh)               # [64,81,256]

# 最后再允许模型混合不同头产生的信息。
attention_output = o_proj(context)                 # [64,81,256]
```

### 6.8 注意力参数是怎样学会“该看哪里”的

训练数据不会提供一张“正确注意力表”。唯一直接监督是下一物品的正确 token。学习链路是：

```text
W_Q/W_K 改变
  → Query 与 Key 的匹配分数改变
  → 注意力权重改变
  → 当前位置从历史读取的内容改变
  → Decoder 的预测概率改变
  → 交叉熵改变
```

如果某次读取方式提高了正确 token 的概率，反向传播会算出如何调整 `W_Q/W_K/W_V/W_O` 才能进一步降低损失；AdamW 再真正修改这些矩阵。经过大量样本反复更新，注意力才逐渐形成有用模式。

所以不能把注意力理解为模型外部维护的规则表。真正持久的是参数；注意力权重只是这些参数遇到当前输入后临时算出的结果。

---

## 7. 一层 Encoder 的完整内部结构

自注意力只解决“不同位置之间怎样交换信息”。一层 Encoder 还必须解决三个问题：

1. 读到信息后，怎样在每个位置内部继续加工？——前馈网络；
2. 堆很多层以后，怎样避免数值和梯度越来越不稳定？——RMSNorm 与残差连接；
3. 怎样降低模型死记训练样本的风险？——Dropout。

一层 T5 Encoder 的主干是：

```text
输入 x
  ↓
RMSNorm → 多头自注意力 → Dropout → 与原 x 相加
  ↓
RMSNorm → 前馈网络     → Dropout → 与上一步相加
  ↓
本层输出
```

### 7.1 注意力和前馈网络不是一回事

可以先用一句话区分：

| 模块 | 主要工作 | 是否让不同位置直接交换信息 |
|---|---|---|
| 自注意力 | 到整段序列中读取相关位置 | 是 |
| 前馈网络 FFN | 对每个位置已经收集到的信息做非线性加工 | 否 |

假设自注意力已经让当前位置收集到“最近连续看了动画片”“部分 token 与儿童类型相关”等信息。此时这些信息仍然只是 256 个混合在一起的数字。前馈网络负责把这种组合进一步变换成对下一层更有用的内部特征。

可以把二者理解成：

```text
注意力：先从其他位置取材料
FFN：再在当前位置加工材料
```

### 7.2 前馈网络最基本的概念

前馈网络 Feed-Forward Network，简称 FFN。这里的“前馈”表示数据从输入经过若干计算流向输出，不在模块内部循环回去。

本项目每个 T5 层中的 FFN 有三步：

```text
256 维输入
  ↓ 第一层线性变换：扩展
1024 维中间特征
  ↓ ReLU：引入非线性
1024 维激活结果
  ↓ 第二层线性变换：压回
256 维输出
```

对应配置：

```text
d_model = 256
d_ff    = 1024
feed_forward_proj = relu
```

#### 什么叫线性变换

线性层的每个输出数字，都是输入数字的加权求和。一个极小例子：

```text
输入 x = [2, -1]

某个输出 = 0.5×2 + 3×(-1) = -2
```

这些 `0.5`、`3` 就是可训练权重。真实模型一次会计算很多个输出，因此把所有权重排成矩阵，并用一次矩阵乘法完成。

#### 为什么先从 256 维扩成 1024 维

扩展不是把同样的数字复制四遍。1024 个中间位置各自拥有不同权重，可以看作 1024 个可学习的“特征响应器”：

```text
某个中间维度可能对一种输入组合响应强
另一个中间维度可能对另一种组合响应强
```

它们的语义不由程序员命名，也不保证能直接解释。更宽的中间层只是给网络更多空间，去表示复杂的特征组合。

#### ReLU 为什么不能省

ReLU 的规则非常简单：

```text
ReLU(x) = max(0, x)
```

例如：

```text
[2.0, -1.0, 0.5] → [2.0, 0.0, 0.5]
```

如果两个线性层之间没有 ReLU，那么“线性变换再接线性变换”在数学上仍可合并为一个线性变换，网络表达能力不会真正增加。ReLU 根据输入把一部分信号保留、一部分归零，使网络能够表示非线性规律。

### 7.3 把一个极小 FFN 完整算出来

假设输入只有 2 维，先扩成 3 维，再压回 2 维：

```text
x = [2, -1]
```

第一层的三个加权组合为：

```text
h₁ = 1×2 + 0×(-1) =  2
h₂ = 0×2 + 1×(-1) = -1
h₃ = 1×2 + 1×(-1) =  1
```

经过 ReLU：

```text
[2, -1, 1] → [2, 0, 1]
```

再压回两维：

```text
y₁ = 0.5×2 + 0×0 + 1×1     =  2.0
y₂ = 0×2   + 1×0 - 0.5×1   = -0.5
```

完整代码如下：

```python
import torch

# 一个位置当前只有两个特征，便于手工核对。
x = torch.tensor([2.0, -1.0])

# 第一层权重：每一行负责产生一个中间特征。
# 形状 [3,2] 表示“2 维输入 → 3 维中间层”。
W_in = torch.tensor([
    [1.0, 0.0],  # 产生 h1
    [0.0, 1.0],  # 产生 h2
    [1.0, 1.0],  # 产生 h3
])

# 矩阵乘法完成三个加权求和。
hidden_before_relu = W_in @ x          # [2,-1,1]

# 负数归零，正数保留，网络由此获得非线性能力。
hidden = torch.relu(hidden_before_relu) # [2,0,1]

# 第二层权重：把 3 维中间特征重新压回 2 维。
W_out = torch.tensor([
    [0.5, 0.0,  1.0],
    [0.0, 1.0, -0.5],
])

# 得到该位置的 FFN 输出。
y = W_out @ hidden                     # [2.0,-0.5]
```

真实 T5 对 batch 中所有位置并行执行同一个 FFN：

```text
输入                  [64, L, 256]
第一层 wi             [64, L, 1024]
ReLU                  [64, L, 1024]
FFN 内部 Dropout       [64, L, 1024]
第二层 wo             [64, L, 256]
```

同一层内，不同位置共享同一组 FFN 参数；不同 Encoder 层则拥有各自独立的 FFN 参数。FFN 不会改变序列长度，也不会把位置 A 与位置 B 相加，跨位置通信已经由注意力完成。

教学等价代码：

```python
import torch

# 两个线性层的权重会训练；T5 这里不使用 bias。
wi = torch.nn.Linear(256, 1024, bias=False)
wo = torch.nn.Linear(1024, 256, bias=False)
dropout = torch.nn.Dropout(p=0.1)

# x 可以包含任意 batch 数和序列长度，最后一维必须是 256。
hidden = wi(x)          # 每个位置：256 → 1024
hidden = torch.relu(hidden)  # 负响应归零，引入非线性
hidden = dropout(hidden)     # 训练时随机屏蔽一部分中间响应
ffn_output = wo(hidden)      # 每个位置：1024 → 256
```

### 7.4 RMSNorm：为什么每个子模块之前都要归一化

层数增加后，向量可能越来越大或越来越小。尺度剧烈变化会让 Softmax、ReLU 和梯度更难稳定训练。RMSNorm 的任务是把一个位置的向量调整到较稳定的整体尺度。

对于向量 `x=[x₁,…,x_D]`，先计算均方根：

```text
RMS(x) = sqrt((x₁² + x₂² + ... + x_D²) / D + ε)
```

再归一化并乘可训练缩放参数 `g`：

```text
RMSNorm(x) = x / RMS(x) × g
```

二维例子，暂时令 `g=[1,1]` 且忽略极小的 `ε`：

```text
x = [3,4]
RMS = sqrt((3²+4²)/2) = sqrt(12.5) ≈ 3.536
归一化结果 ≈ [0.849,1.131]
```

`g` 是 256 维可训练参数，允许模型在归一化后重新决定各维度的合适尺度。

T5 的 RMSNorm 与常见 LayerNorm 有一个重要差别：它不先减去均值，主要依据均方根调整尺度。归一化不会“理解电影语义”，它只是帮助后续计算处于更稳定的数值范围。

下面是可运行的最小实现。第 7.7 节中的 `RMSNorm` 指的就是这类模块：

```python
import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, dimension, eps=1e-6):
        super().__init__()

        # 每个维度有一个可训练缩放值；初始都设为 1。
        self.weight = torch.nn.Parameter(torch.ones(dimension))

        # eps 防止输入全为 0 时出现除以 0。
        self.eps = eps

    def forward(self, x):
        # 对最后一维求平方平均。
        # keepdim=True 保留一个长度为 1 的维度，便于广播。
        mean_square = x.float().pow(2).mean(dim=-1, keepdim=True)

        # rsqrt(a) 等于 1/sqrt(a)，因此这里得到归一化系数。
        inverse_rms = torch.rsqrt(mean_square + self.eps)

        # 先在 float32 中调整整体尺度，减少低精度下的数值误差。
        normalized = x.float() * inverse_rms

        # 再对齐到可训练 weight 的 dtype，并逐维缩放。
        normalized = normalized.to(dtype=self.weight.dtype)
        return normalized * self.weight
```

### 7.5 残差连接：为什么要把输入再加回来

若只写：

```text
新 x = Attention(RMSNorm(x))
```

那么每一层都必须完整重写输入表示。残差连接改成：

```text
新 x = 原 x + Attention(RMSNorm(原 x))
```

这意味着子模块更像是在学习“应该给原表示增加什么修正量”。如果某个子模块暂时学不到有用内容，只要输出接近 0，原信息仍能沿加法路径继续向后传递。

残差连接对反向传播也重要。梯度可以沿着加法中的直接路径传播，不必每次都穿过复杂的注意力或 FFN，深层网络因此更容易训练。

注意：残差相加要求两边形状一致。这也是注意力和 FFN 最终都必须输出 256 维的原因。

### 7.6 Dropout：训练时故意制造一点不确定性

Dropout 在训练时随机把部分中间数字置零，其余数字做相应缩放。例如 `p=0.1` 表示每个元素有 10% 概率被屏蔽。

作用不是节省计算，而是减少模型对某一条固定内部路径的依赖，降低死记训练样本的风险：

```text
训练：每次前向使用的随机屏蔽模式可能不同
验证/推理：关闭随机屏蔽，使用完整网络
```

`model.train()` 启用 Dropout，`model.eval()` 关闭 Dropout。Dropout 自身没有需要学习的参数。

### 7.7 一层 Encoder 的逐行教学代码

下面把第 6 节的注意力模块和本节的 FFN 连接起来。它是结构等价的教学代码，不包含 Hugging Face 的缓存、返回值封装等工程细节：

```python
class EncoderBlock(torch.nn.Module):
    def __init__(self, self_attention, d_model=256, d_ff=1024, p=0.1):
        super().__init__()

        # 自注意力模块内部保存 W_Q、W_K、W_V、W_O。
        self.self_attention = self_attention

        # 两个 RMSNorm 分别服务于注意力和 FFN。
        # 它们各自拥有一组可训练的缩放参数。
        self.norm_before_attention = RMSNorm(d_model)
        self.norm_before_ffn = RMSNorm(d_model)

        # FFN：先扩展到 1024 维，再压回 256 维。
        self.wi = torch.nn.Linear(d_model, d_ff, bias=False)
        self.wo = torch.nn.Linear(d_ff, d_model, bias=False)

        # Dropout 没有可训练权重，只在训练模式随机屏蔽。
        self.dropout = torch.nn.Dropout(p)

    def forward(self, x, attention_mask, position_bias):
        # ---------- 子层 1：自注意力 ----------
        residual = x
        # T5 使用 Pre-Norm：先归一化，再进入复杂子模块。
        normalized = self.norm_before_attention(x)

        # 所有真实历史位置彼此读取；PAD 由 mask 排除。
        attended = self.self_attention(
            normalized,
            attention_mask=attention_mask,
            position_bias=position_bias,
        )

        # 保留原输入，并加入注意力学到的改变量。
        x = residual + self.dropout(attended)

        # ---------- 子层 2：前馈网络 ----------
        residual = x
        normalized = self.norm_before_ffn(x)

        # 每个位置独立执行 256 → 1024 → 256。
        hidden = self.wi(normalized)
        hidden = torch.relu(hidden)
        hidden = self.dropout(hidden)
        transformed = self.wo(hidden)

        # 再次通过残差连接保留子层输入。
        x = residual + self.dropout(transformed)
        return x
```

真实库中的类名和代码组织不同，但数据流就是：

```python
x = x + dropout(self_attention(rms_norm(x)))
x = x + dropout(ffn(rms_norm(x)))
```

### 7.8 六层 Encoder 到底逐层做了什么

本项目连续堆叠 6 层：

```text
token Embedding [64,L,256]
  ↓ Encoder layer 1：交换信息 + 每位置加工
  ↓ Encoder layer 2：在已有上下文上再次交换与加工
  ↓ Encoder layer 3
  ↓ Encoder layer 4
  ↓ Encoder layer 5
  ↓ Encoder layer 6
  ↓ 最终 RMSNorm
历史 memory [64,L,256]
```

每层输入输出形状相同，但参数不共享。第 2 层看到的已经不是原始 token Embedding，而是第 1 层结合上下文后的表示；越往后，网络可以在前层结果上继续组合更复杂的模式。

不能武断地说“第 1 层只学电影类型，第 6 层只学用户偏好”。训练没有这种逐层标签。能够确定的是计算角色：每层都包含跨位置读取和逐位置非线性加工。

最终 `memory` 不是把整段历史压成一个向量，而是为每个输入位置保留一个 256 维上下文表示：

```text
memory[样本, 历史位置, 256维上下文表示]
```

这里的“memory”只是 Encoder 本次前向产生、供 Decoder 查询的张量，不是数据库，也不是跨用户永久保存的记忆。真正长期保存的知识仍然是模型参数。

---

## 8. Decoder：怎样一边读取历史，一边写出下一物品

Encoder 和 Decoder 的分工是整个模型的核心：

```text
Encoder：把用户历史读成一组上下文表示
Decoder：根据历史表示和已经写出的答案前缀，继续写下一个 token
```

Encoder 像“读者”，一次看到完整的已知历史；Decoder 像“作者”，输出端的未来答案不存在，只能从左到右生成。

### 8.1 为什么只有 Encoder 还不能完成生成

Encoder 的输出是：

```text
memory [64, L, 256]
```

它描述每个历史位置在上下文中的含义，但没有直接规定：

- 下一物品第一位语义码是什么；
- 已经生成第一位后，第二位应该如何受它影响；
- 什么时候输出 EOS。

Decoder 专门维护“答案目前写到哪里”，并在每一步同时读取：

1. 已经生成的答案前缀；
2. Encoder 编好的用户历史。

因此，Decoder 不是 Encoder 的重复版。它比 Encoder 多了“读取 Encoder”的交叉注意力，还必须用因果 mask 防止偷看未来答案。

### 8.2 训练前先把标签右移一位

假设某条训练样本的正确目标 token 是：

```text
[25, 494, 519, 771, EOS]
```

模型需要预测这五个 token。若把正确 token 原样放在对应输入位置，模型就能直接看到答案。T5 因而自动右移标签，并在开头加入 BOS：

```text
正确标签：  [25,  494, 519, 771, EOS]
Decoder输入：[BOS, 25,  494, 519, 771]
```

对齐后形成五道题：

| 输出位置 | Decoder 当前可用的正确前缀 | 本位置应预测 |
|---:|---|---:|
| 0 | `BOS` | 25 |
| 1 | `BOS, 25` | 494 |
| 2 | `BOS, 25, 494` | 519 |
| 3 | `BOS, 25, 494, 519` | 771 |
| 4 | `BOS, 25, 494, 519, 771` | EOS |

训练时给 Decoder 正确的历史答案前缀，称为 **teacher forcing**。它不是把当前答案直接泄漏给当前预测位置：位置 2 的输入只能包含位置 0、1 的正确 token，位置 2 的答案仍然需要预测。

### 8.3 一层 Decoder 为什么有三段计算

每层 Decoder 依次执行：

```text
第一段：因果自注意力
        回答“答案前缀已经写了什么？”

第二段：交叉注意力
        回答“为了继续写，应该从用户历史读取什么？”

第三段：前馈网络
        回答“怎样加工刚刚汇总的信息？”
```

三段缺一不可：

- 没有因果自注意力，生成第二位时不知道第一位是什么；
- 没有交叉注意力，不同用户历史可能得到几乎一样的输出；
- 没有 FFN，每个位置只能做注意力产生的加权混合，缺少进一步的非线性加工。

每段外面都使用与 Encoder 类似的 `RMSNorm → 子模块 → Dropout → 残差相加`。Decoder 只是比 Encoder 多一个交叉注意力子层。

### 8.4 第一段：因果自注意力究竟在做什么

Decoder 自注意力的 Q、K、V 全部来自 Decoder 当前状态：

```text
Q_dec = decoder_states × W_Q_self
K_dec = decoder_states × W_K_self
V_dec = decoder_states × W_V_self
```

这与 Encoder 自注意力一样，也是“同一个序列内部互相读取”。区别在于 Decoder 必须遮住未来位置。

对五个输出位置，因果 mask 允许读取的范围如下：

| 查询位置 | 能读取 Decoder 输入位置 | 不能读取 |
|---:|---|---|
| 0 | 0（BOS） | 1、2、3、4 |
| 1 | 0、1 | 2、3、4 |
| 2 | 0、1、2 | 3、4 |
| 3 | 0、1、2、3 | 4 |
| 4 | 0、1、2、3、4 | 无 |

写成矩阵，`✓` 表示允许注意，`×` 表示分数在 Softmax 前被 mask：

```text
          被读取位置
查询位置    0  1  2  3  4
   0       ✓  ×  ×  ×  ×
   1       ✓  ✓  ×  ×  ×
   2       ✓  ✓  ✓  ×  ×
   3       ✓  ✓  ✓  ✓  ×
   4       ✓  ✓  ✓  ✓  ✓
```

例如，预测第三位 token `519` 的状态能够读取：

```text
BOS、25、494
```

但不能读取 Decoder 输入中后面的 `519、771`。因此它知道已经选择的前两位语义码，却看不到自己要预测的答案。

真实形状为：

```text
Decoder 状态              [64, 5, 256]
8 头 Q/K/V                [64, 8, 5, 32]
因果自注意力分数           [64, 8, 5, 5]
自注意力输出               [64, 5, 256]
```

这一段的参数属于 Decoder 本层自己，与 Encoder 自注意力参数不共享。

带注释的教学代码：

```python
# decoder_states 是右移标签经过 Embedding 或上一层后的表示。
q = decoder_self_q(decoder_states)  # [B,5,256]
k = decoder_self_k(decoder_states)  # [B,5,256]
v = decoder_self_v(decoder_states)  # [B,5,256]

# 拆成 8 个头；每个头独立处理 32 维。
q = split_heads(q)                  # [B,8,5,32]
k = split_heads(k)                  # [B,8,5,32]
v = split_heads(v)                  # [B,8,5,32]

# 五个输出位置两两比较，暂时得到完整的 5×5 分数表。
scores = q @ k.transpose(-2, -1)    # [B,8,5,5]

# causal_mask 的上三角为 False。
# 未来位置被填成极小值，Softmax 后权重接近 0。
scores = scores.masked_fill(
    ~causal_mask,
    torch.finfo(scores.dtype).min,
)

# 每个位置只在自己和左侧前缀之间分配读取比例。
weights = torch.softmax(scores, dim=-1)
self_context = weights @ v          # [B,8,5,32]
```

### 8.5 第二段：交叉注意力究竟在做什么

完成第一段后，Decoder 已经知道“答案前缀写了什么”，但还需要查询用户历史。交叉注意力连接 Decoder 与 Encoder：

```text
Query 来自 Decoder：当前生成位置想从历史找什么
Key 来自 Encoder：每个历史位置怎样参与匹配
Value 来自 Encoder：匹配后每个历史位置提供什么内容
```

公式为：

```text
Q_cross = decoder_states × W_Q_cross
K_cross = encoder_memory × W_K_cross
V_cross = encoder_memory × W_V_cross
```

为什么 Query 来自 Decoder？因为不同生成位置的需求不同：

- 生成第一位时，可能更需要粗粒度偏好；
- 已经生成前三位后，第四位主要用于完成唯一标识；
- 生成 EOS 时，需要判断四位码是否已经完整。

为什么 Key 和 Value 来自 Encoder？因为被检索的对象是用户历史，而 Encoder 已经把每个历史位置加工成上下文化表示。

#### 一个最小的交叉注意力例子

假设 Decoder 当前位置产生查询：

```text
Q = [1,0]
```

Encoder 的三个历史位置产生：

```text
K_A = [0,1]    V_A = [1,0]
K_B = [1,0]    V_B = [0,2]
K_C = [2,0]    V_C = [1,1]
```

匹配分数：

```text
Q·K_A = 0
Q·K_B = 1
Q·K_C = 2
```

Softmax 后：

```text
[0.090,0.245,0.665]
```

Decoder 从用户历史读取到：

```text
0.090×V_A + 0.245×V_B + 0.665×V_C
= [0.755,1.155]
```

这与自注意力使用同一套“Q/K 匹配，再加权 V”的数学机制。真正的差别不是公式，而是数据来源。

真实形状：

```text
Decoder Query             [64, 8, 5, 32]
Encoder Key/Value         [64, 8, L, 32]
交叉注意力分数             [64, 8, 5, L]
读取历史后的结果           [64, 8, 5, 32]
拼接并投影                [64, 5, 256]
```

交叉注意力不使用 Decoder 的三角因果 mask，因为整个 Encoder 输入都是已知历史；它只使用 Encoder attention mask 排除 PAD。

交叉注意力拥有自己的 `W_Q_cross/W_K_cross/W_V_cross/W_O_cross`，不与 Encoder 自注意力或 Decoder 因果自注意力共享。

带注释的教学代码：

```python
# 当前 Decoder 状态提出查询。
q = cross_q(decoder_states)          # [B,5,256]

# Encoder memory 为每个历史位置提供匹配 Key 和内容 Value。
k = cross_k(encoder_memory)          # [B,L,256]
v = cross_v(encoder_memory)          # [B,L,256]

# 分成 8 个头。Decoder 长度是 5，Encoder 长度是 L。
q = split_heads(q)                   # [B,8,5,32]
k = split_heads(k)                   # [B,8,L,32]
v = split_heads(v)                   # [B,8,L,32]

# 每个输出位置分别与全部历史位置匹配。
scores = q @ k.transpose(-2, -1)     # [B,8,5,L]

# Encoder 中的 PAD 不能被读取。
scores = scores.masked_fill(
    ~encoder_mask[:, None, None, :],
    torch.finfo(scores.dtype).min,
)

# 在 L 个历史位置上分配读取比例，并混合 Encoder Value。
weights = torch.softmax(scores, dim=-1)
history_context = weights @ v        # [B,8,5,32]

# 拼回 256 维，再经过交叉注意力自己的输出投影。
history_context = merge_heads(history_context)  # [B,5,256]
history_context = cross_o(history_context)       # [B,5,256]
```

### 8.6 第三段：Decoder FFN 在做什么

经过前两段，某个输出位置已经拥有：

```text
答案前缀提供的信息
+ 用户历史提供的信息
```

但这些信息仍然主要由注意力加权混合得到。FFN 对每个输出位置独立执行：

```text
256 → 1024 → ReLU → 256
```

它与第 7 节 Encoder FFN 的机制相同，但参数是 Decoder 本层自己的。它负责把“前缀信息 + 历史信息”的组合加工成更适合下一层或最终输出分类的表示。

### 8.7 一层 Decoder 的完整带注释代码

下面把三段连接起来：

```python
class DecoderBlock(torch.nn.Module):
    def forward(
        self,
        decoder_states,       # [B,5,256]：答案前缀的当前表示
        encoder_memory,       # [B,L,256]：Encoder 编好的用户历史
        causal_mask,          # 防止读取未来答案
        encoder_mask,         # 防止读取历史 PAD
    ):
        # ===== 1. 因果自注意力：读取已生成的答案前缀 =====
        residual = decoder_states
        x = self.norm_before_self_attention(decoder_states)
        x = self.causal_self_attention(
            x,
            causal_mask=causal_mask,
        )
        decoder_states = residual + self.dropout(x)

        # ===== 2. 交叉注意力：用当前输出状态查询用户历史 =====
        residual = decoder_states
        x = self.norm_before_cross_attention(decoder_states)
        x = self.cross_attention(
            query_states=x,              # Q 来自 Decoder
            key_value_states=encoder_memory,  # K/V 来自 Encoder
            encoder_mask=encoder_mask,
        )
        decoder_states = residual + self.dropout(x)

        # ===== 3. FFN：逐位置加工前缀信息与历史信息 =====
        residual = decoder_states
        x = self.norm_before_ffn(decoder_states)
        x = self.wi(x)                    # 256 → 1024
        x = torch.relu(x)                 # 引入非线性
        x = self.ffn_dropout(x)            # 训练时随机屏蔽
        x = self.wo(x)                    # 1024 → 256
        decoder_states = residual + self.dropout(x)

        return decoder_states
```

本项目堆叠 6 层 Decoder。每一层都重新执行“读前缀 → 读历史 → 逐位置加工”，且每层参数独立。

### 8.8 为什么训练时五个位置可以同时计算

从逻辑上看，第二位依赖第一位，第三位依赖前两位，似乎必须逐个运行。但训练时：

1. 正确前缀已通过右移标签放进 Decoder 输入；
2. 因果 mask 严格阻止每个位置读取未来；
3. 矩阵运算可以同时计算五行，只是每行允许读取的范围不同。

因此一次前向可以并行得到：

```text
Decoder 输入 token         [64,5]
共享 Embedding 查表后       [64,5,256]
6 层 Decoder 后            [64,5,256]
输出层 logits              [64,5,1027]
```

“同时计算”不等于“互相看见未来”。因果 mask 把不允许的连接切断了。

### 8.9 为什么推理时又必须逐个生成

推理时没有正确标签，开始只有：

```text
[BOS]
```

过程是：

```text
第 1 次前向：[BOS]              → 预测第 1 位 token
第 2 次前向：[BOS,第1位]         → 预测第 2 位 token
第 3 次前向：[BOS,第1位,第2位]   → 预测第 3 位 token
第 4 次前向：加入第3位           → 预测第 4 位 token
第 5 次前向：加入第4位           → 预测 EOS
```

后一步的输入包含前一步模型自己选出的 token，所以无法在第一次前向前就知道全部输入。`model.generate()` 会自动管理这一循环，并缓存已经算过的 Decoder Key/Value，避免每一步从头重复所有前缀计算。

### 8.10 用一个物品理解 Decoder 五个位置的分工

以标签 `[25,494,519,771,EOS]` 为例：

```text
位置 0：看到 BOS
       → 因果自注意力只能读 BOS
       → 交叉注意力查询用户历史
       → 预测第一位 token 25

位置 1：看到 BOS、25
       → 因果自注意力知道第一位已经选了 25
       → 再查询用户历史
       → 预测第二位 token 494

位置 2：看到 BOS、25、494
       → 结合已有两位和用户历史
       → 预测第三位 token 519

位置 3：看到前三位
       → 结合完整量化前缀与历史
       → 预测消歧 token 771

位置 4：看到完整四位 token
       → 判断一个物品标识符已经结束
       → 预测 EOS
```

这里“第一位更粗、后续位逐步补充残差”来自语义 ID 的构造方式，但不能据此断言某个具体注意力头一定只看类型或某部电影。注意力究竟集中在哪里，需要实际导出权重才能分析。

### 8.11 Encoder 与 Decoder 中到底有哪些参数

初学时最容易把临时张量和模型参数混在一起。下面统一整理：

| 部位 | 长期训练参数 | 本次前向临时产生 |
|---|---|---|
| 共享 Embedding | `1027×256` 的向量表 | 输入 token 对应的向量 |
| 每层 Encoder 自注意力 | `W_Q/W_K/W_V/W_O` | Encoder Q/K/V、分数、权重 |
| 每层 Encoder FFN | `W_in/W_out` | 1024 维中间激活 |
| 每层 Decoder 自注意力 | 独立的 `W_Q/W_K/W_V/W_O` | 前缀 Q/K/V、因果权重 |
| 每层 Decoder 交叉注意力 | 独立的 `W_Q/W_K/W_V/W_O` | Decoder Q、Encoder K/V、交叉权重 |
| 每层 Decoder FFN | 独立的 `W_in/W_out` | 1024 维中间激活 |
| RMSNorm | 每个 Norm 的缩放参数 | 当前归一化结果 |
| 输出层 | 与共享 Embedding 绑定的输出权重 | 1,027 个 logits |

模型训练时，最终交叉熵的梯度会穿过 Decoder、交叉注意力和 Encoder，更新上述全部相关参数。Encoder 并不是先单独训完再冻结；在当前 T5 训练中，Encoder 与 Decoder 从随机初始化开始联合训练。

### 8.12 这些计算在项目脚本的哪一行发生

仓库没有重新手写一套 T5，而是在 [`build_model()`](../scripts/train_generative.py) 中创建 Hugging Face 的 `T5ForConditionalGeneration`：

```python
# 这些配置决定前文所有真实形状。
model_cfg = T5Config(
    vocab_size=1027,          # 输出候选 token 数
    d_model=256,              # 每个 token 的主表示维度
    d_ff=1024,                # 每层 FFN 的中间宽度
    num_layers=6,             # Encoder 层数
    num_decoder_layers=6,     # Decoder 层数
    num_heads=8,              # 每个注意力模块的头数
    d_kv=32,                  # 每个头的 Q/K/V 维度
    dropout_rate=0.1,
    feed_forward_proj="relu",
    pad_token_id=0,
    decoder_start_token_id=1, # BOS
    eos_token_id=2,
)

# 这一行真正创建共享 Embedding、6 层 Encoder、6 层 Decoder、
# 每层内部的 Q/K/V/O 投影、FFN、RMSNorm 和最终输出层。
model = T5ForConditionalGeneration(model_cfg)
```

训练循环中的真实入口只有：

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    # batch 含 input_ids、attention_mask 和 labels。
    # 调用这一行时，T5 内部依次完成：
    #   1. Encoder Embedding 与 6 层 Encoder；
    #   2. labels 右移，构造 Decoder 输入；
    #   3. 6 层 Decoder；
    #   4. 输出 1,027 维 logits；
    #   5. 与 labels 计算交叉熵。
    raw_loss = model(**batch).loss
```

所以脚本中看不到显式的 `Q = X @ W_Q`，并不表示项目没有 Q/K/V；这些参数和运算封装在 `T5ForConditionalGeneration` 内部。仓库负责冻结正确配置、准备正确张量并调用模型，Transformers 库负责执行标准 T5 层。

反向传播时：

```python
loss.backward()
```

PyTorch 会继续穿过这层库封装，为内部所有 Q/K/V、FFN、Embedding 等参数计算梯度。随后：

```python
optimizer.step()
```

才会真正修改这些参数。封装只减少了仓库中的重复代码，没有改变前文描述的训练机制。

### 8.13 把 Encoder–Decoder 的灵魂压缩成一条链

```text
历史 token
  ↓ 共享 Embedding
孤立的 256 维位置向量
  ↓ 6 层 Encoder：历史内部反复交流和加工
上下文化历史 memory
  ↓
右移后的答案前缀 → 6 层 Decoder
                    ├─ 因果自注意力：读已经写出的答案
                    ├─ 交叉注意力：按当前需求读取历史 memory
                    └─ FFN：加工两类信息
  ↓
每个输出位置的 256 维状态
  ↓ 输出层
每个位置对 1,027 个 token 的 logits
```

Encoder 决定“历史被怎样理解”，Decoder 决定“在当前已生成前缀下，应从历史取什么并继续写什么”。两部分通过交叉注意力相连，这正是 Encoder–Decoder 生成模型最关键的接口。

---

## 9. 输出层：256 维状态怎样变成 1,027 个候选分数

### 9.1 三分类最小例子

假设 Decoder 当前只产生一个数字 `h=2`，三个候选的输出权重是：

```text
候选甲： 0.5
候选乙： 0.0
候选丙：-0.5
```

三个原始分数是：

```text
[2×0.5, 2×0.0, 2×(-0.5)] = [1,0,-1]
```

Softmax 转换后：

```text
甲：66.5%
乙：24.5%
丙： 9.0%
```

原始分数叫 logits。训练计算交叉熵时直接使用 logits；交叉熵内部会以数值稳定的方式完成 Softmax 与对数计算。

### 9.2 真实张量规模

输出层对 Decoder 每个位置都产生 1,027 个 logits：

```text
Decoder 状态 [64,5,256]
      ↓ lm_head
logits        [64,5,1027]
```

这表示 64 个样本、每个样本 5 个预测位置、每个位置 1,027 个候选分数。

训练时没有按位置手工屏蔽词表，模型通过标签学习第 1 位应使用第一段 token、第 2 位应使用第二段 token。推理时则由 Trie 显式保证位置与目录路径合法。

---

## 10. 损失：怎样把“预测得有多差”压成一个数

### 10.1 一个位置的最小例子

如果正确答案是候选乙，而模型给乙的概率为 24.5%，交叉熵是：

```text
loss = -ln(0.245) ≈ 1.408
```

正确答案概率与损失的关系：

| 正确答案概率 | 损失 `-ln(p)` |
|---:|---:|
| 1% | 4.605 |
| 10% | 2.303 |
| 50% | 0.693 |
| 90% | 0.105 |

正确答案概率越低，惩罚越大；概率越接近 1，损失越接近 0。相比只记录“对/错”，连续损失能区分“正确答案概率从 1% 提升到 20%”这样的进步，并且可以求导。

### 10.2 真实 batch 的损失

一个 micro-batch 有：

```text
64 个样本 × 每个 5 个目标位置 = 320 个预测位置
```

概念上，程序计算 320 个正确 token 的负对数概率并取平均。标签中的 `-100` 位置不参与平均。

T5 内部等价于：

```python
loss = CrossEntropyLoss(ignore_index=-100)(
    logits.view(-1, 1027),
    labels.view(-1),
)
```

项目中的：

```python
raw_loss = model(**batch).loss
```

一行完成了从 Embedding 到这个标量损失的全部前向计算。

---

## 11. 反向传播：一个错误如何传回 1,128 万个参数

### 11.1 先理解梯度

某个参数的梯度回答：

```text
在其他量暂时不变时，参数稍微增大，loss 会向哪个方向变化、变化多快？
```

- 梯度为正：增大参数会增大损失，为降低损失应向小调整；
- 梯度为负：增大参数会降低损失，为降低损失应向大调整；
- 绝对值越大：当前位置附近，损失对该参数越敏感。

梯度只描述当前位置附近的局部趋势，不保证沿同一方向无限走都有效。

### 11.2 可以直接运行的最小训练程序

```python
import torch
import torch.nn.functional as F

h = torch.tensor([[2.0]])
weight = torch.nn.Parameter(
    torch.tensor([[0.5], [0.0], [-0.5]])
)
target = torch.tensor([1])
optimizer = torch.optim.SGD([weight], lr=0.1)

# 前向传播
logits = h @ weight.T
loss = F.cross_entropy(logits, target)

print(torch.softmax(logits, dim=-1))
# [[0.6652, 0.2447, 0.0900]]

# 只计算梯度，参数尚未改变
loss.backward()
print(weight.grad)
# [[ 1.3305], [-1.5105], [0.1801]]

# 真正更新参数
optimizer.step()

with torch.no_grad():
    new_logits = h @ weight.T
    print(torch.softmax(new_logits, dim=-1))
    # [[0.5495, 0.3568, 0.0936]]
```

一次更新后，正确候选 1 的概率从 `24.47%` 上升到 `35.68%`，损失从 `1.4076` 下降到 `1.0305`。

关键区别：

```text
loss.backward()  → 计算并累积 parameter.grad，参数本身不变
optimizer.step() → 根据 grad 真正修改 parameter
```

### 11.3 梯度怎样穿过整个 T5

前向传播建立了计算关系：

```text
Embedding 参数
  → Encoder 的 Q/K/V、前馈网络
  → 历史 memory
  → Decoder 自注意力、交叉注意力、前馈网络
  → lm_head logits
  → cross entropy loss
```

反向传播按相反方向应用链式法则：

```text
loss
  → logits
  → lm_head
  → Decoder
  → Cross-Attention
  → Encoder
  → Embedding
```

如果一个参数通过多条路径影响 loss，各条路径的梯度会相加。PyTorch 在前向时记录实际执行过的张量运算，`backward()` 自动沿这张计算图反向计算，不需要把每个参数分别扰动后重跑模型。

因此训练没有收到“请让第 3 个头多看最近一部电影”这种人工规则。它只收到正确下一物品。某种注意力方式若经常提高正确 token 的概率，对应参数就会从大量样本的梯度中得到强化。

---

## 12. 四个 micro-batch 如何完成一次参数更新

### 12.1 项目真实训练循环

[`train_generative.py`](../scripts/train_generative.py) 的核心逻辑：

```python
optimizer.zero_grad(set_to_none=True)

for _ in range(accumulation):            # accumulation = 4
    batch = next(iterator)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        raw_loss = model(**batch).loss
        loss = raw_loss / accumulation

    loss.backward()

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)

optimizer.step()
scheduler.step()
optimizer.zero_grad(set_to_none=True)
step += 1
```

按执行顺序理解：

```text
micro-batch 1：64 样本 → loss/4 → backward → 累积梯度
micro-batch 2：64 样本 → loss/4 → backward → 累积梯度
micro-batch 3：64 样本 → loss/4 → backward → 累积梯度
micro-batch 4：64 样本 → loss/4 → backward → 累积梯度
                                      ↓
                                梯度裁剪
                                      ↓
                            AdamW 更新一次参数
                                      ↓
                           调整学习率并清空梯度
```

有效 batch size 为：

```text
64 × 4 = 256 个样本/optimizer step
```

除以 4 是为了让累计结果接近 256 个样本的平均梯度，而不是把梯度整体放大四倍。

### 12.2 AdamW 在做什么

最简单的梯度下降是：

```text
新参数 = 旧参数 - 学习率 × 当前梯度
```

AdamW 还保存每个参数过去梯度的一阶、二阶移动统计，用它们调整当前更新尺度，并独立施加权重衰减。项目设置为：

```text
优化器：AdamW
峰值学习率：3e-4
weight decay：0.01
梯度范数上限：1.0
```

学习率不是“每个参数固定减去 0.0003”。实际改变量还取决于该参数的梯度和 AdamW 历史状态。

### 12.3 学习率日程

正式训练最多 30,000 optimizer steps：

```text
step 0～1500：学习率线性升到 3e-4
step 1500～30000：学习率线性降到 0
```

warmup 避免随机初始化模型刚开始就进行过大的更新；后期衰减让参数在已经较好的区域内做更小调整。

### 12.4 BF16、梯度裁剪与 train 模式

- BF16 autocast 让大量矩阵运算以较低精度执行，降低显存和计算成本；
- 梯度裁剪在梯度整体过大时等比例缩小，避免单次异常更新；
- `model.train()` 会启用 Dropout 等训练行为，但它本身不会更新参数；
- 真正修改参数的是 `optimizer.step()`。

---

## 13. step、epoch 与模型已经看过多少数据

981,491 条训练样本、micro-batch 64：

```text
一个 epoch ≈ ceil(981491 / 64) = 15,336 个 micro-batch
```

每 4 个 micro-batch 更新一次：

```text
一个 epoch ≈ 15,336 / 4 = 3,834 optimizer steps
```

所以：

```text
30,000 steps ≈ 7.82 epochs
```

程序以 optimizer step 为停止条件。DataLoader 走完一轮后重新打乱样本并继续，所以 epoch 客观存在，只是没有作为主日志单位。

每次重新看到同一训练题时，模型参数已改变，Dropout 也可能不同，产生的预测、损失与梯度一般不会完全相同。

---

## 14. 验证与推理：没有正确答案以后怎样生成

### 14.1 训练和推理的关键区别

训练时：

```text
Decoder 前缀来自正确标签右移，五个位置可以并行算。
```

推理时：

```text
只有 BOS；第 2 步必须等待第 1 步生成结果，第 3 步再等待前两步。
```

因此推理要自回归运行 5 步：四位物品 token 加 EOS。

验证使用：

```python
model.eval()

with torch.inference_mode():
    generated = model.generate(...)
```

`eval()` 关闭 Dropout；`inference_mode()` 不记录反向计算图，节省时间和显存。它们都不会修改模型参数。

### 14.2 为什么不能每步只取最高概率

假设第一步：

```text
路径 A：概率 0.51
路径 B：概率 0.49
```

贪心会立刻丢弃 B。但下一步可能是：

```text
A 的最佳延续概率 0.10 → 完整路径约 0.051
B 的最佳延续概率 0.90 → 完整路径约 0.441
```

第一步稍低的 B 最终更好。Beam Search 因此保留多条部分路径。

项目设置：

```text
num_beams = 20
num_return_sequences = 10
max_new_tokens = 5
```

每一步扩展当前候选路径，按累计序列得分保留 20 条，最后返回 10 条完整路径。

---

## 15. Trie 约束：为什么生成结果一定能翻译成真实电影

### 15.1 最小示例

假设目录只有三个合法物品：

```text
[10,20,30,0]
[10,20,31,0]
[11,40,50,0]
```

Trie 是：

```text
根
├── 10
│   └── 20
│       ├── 30
│       │   └── 0 → 物品 A
│       └── 31
│           └── 0 → 物品 B
└── 11
    └── 40
        └── 50
            └── 0 → 物品 C
```

如果已经生成 `[10,20]`，下一步只允许 30 或 31。即使模型给其他 token 很高分，也不能走向不存在的物品。

### 15.2 动态排除已见物品

项目 Trie 不只检查目录合法性，还保存每个前缀能够通向哪些物品。

若用户已经看过物品 A，那么 `[10,20,30]` 分支被排除；只要 `[10,20,31]` 仍通向未见物品 B，前缀 `[10,20]` 就仍然合法。

生成四位后，只在叶子对应真实且未见物品时允许 EOS。

Encoder 只读取最近 20 部电影，但过滤已见物品使用完整历史，因此不会因为输入截断而推荐更早看过的电影。

### 15.3 对应项目代码

[`SemanticTrie.allowed()`](../scripts/generative_common.py) 的核心逻辑：

```python
for token in self.children.get(prefix, ()):
    child = prefix + (token,)
    if self.descendants.get(child, set()).difference(seen_items):
        allowed.append(token)
```

生成时把它交给 Hugging Face：

```python
generated = model.generate(
    input_ids=batch["input_ids"].to(device),
    attention_mask=batch["attention_mask"].to(device),
    num_beams=20,
    num_return_sequences=10,
    max_new_tokens=5,
    prefix_allowed_tokens_fn=prefix_fn,
)
```

Stage 2–3 的约束验证中，目录非法率、已见物品重复率和短列表率均为 0，说明该机制按预期工作。

---

## 16. 指标、选模与 checkpoint

### 16.1 单用户最小例子

每名用户只有一个下一物品目标。若目标排在第 2 位：

```text
Recall@5  = 1
Recall@10 = 1
NDCG@5    = 1/log2(2+1) ≈ 0.631
NDCG@10   = 0.631
```

如果目标不在 Top-10，四项均为 0。最后对所有用户求平均。

Recall 只判断是否命中；NDCG 还奖励更靠前的排名。

### 16.2 真实验证流程

正式训练每 5,000 optimizer steps 对完整 6,040 名 valid 用户执行约束生成，并使用整体 NDCG@10 选模：

```text
当前 NDCG@10 高于历史最佳 → 保存 best，bad_rounds 清零
没有提高                   → bad_rounds 加一
连续 3 次没有提高           → 早停
```

每次验证还保存 `last`。二者含义不同：

- `best`：截至当前验证指标最好的模型；
- `last`：最近验证点的模型，主要用于恢复训练。

checkpoint 包含模型参数、optimizer、scheduler、global step 和主要随机数状态。当前恢复机制保证训练状态可以继续，但没有保存 DataLoader sampler 的精确游标，因此不承诺中断前后逐 batch、逐 bit 完全一致。

---

## 17. Stage 0–3 为什么要在正式训练前完成

### Stage 0：冻结并审计接口

检查物品数、用户数、训练样本数、四位 ID 唯一性、token 双向转换、词表大小和模型参数量。它回答“数据是否送对了”。

### Stage 1：实现完整训练与评测程序

实现数据集、模型、checkpoint、Beam Search、Trie、指标和结果汇总。它回答“所有程序组件是否齐全”。

### Stage 2：验证程序真的能学习

- smoke test 验证前向、反向、保存与生成；
- 从 step 20 恢复到 step 30，验证断点续训；
- 256 样本过拟合把 loss 从约 3.957 降到 0.001613，证明标签、梯度和优化器链路连通。

小样本过拟合的 valid 指标不代表模型最终能力；它只验证模型能否记住训练数据。如果连很小的数据都学不会，正式训练通常没有意义。

### Stage 3：控制变量选择学习率

固定 seed、数据、步数和验证用户，只比较 `3e-4` 与 `1e-3`。step 5000 的 valid NDCG@10 分别为：

```text
3e-4：0.056145
1e-3：0.039042
```

因此正式三种子训练冻结 `3e-4`。pilot 只用于选超参数，不访问 test，也不能代替最终三种子结果。

---

## 18. 用真实电影把训练与验证全链路再走一遍

训练和验证不能混为一谈：用户 0 的 `Mulan (1998)` 是 **valid 目标**，不会作为该用户的训练标签泄漏给模型。下面先用其训练序列中的电影说明参数如何学习，再用 Mulan 说明模型如何接受检验。

### 18.1 训练：模型怎样从一个已知目标学习

以训练序列中 `Toy Story (1995)` 的某个前缀样本为例：

1. 数据集取目标之前的历史，最多保留最近 20 部电影。
2. 每部历史电影的四位 ID 转成四个位置 token，展平后追加 Encoder EOS。
3. `TrainCollator` 将 64 条不同长度历史补 PAD，产生 `input_ids` 与 `attention_mask`。
4. Embedding 将 `[64,L]` 个整数变为 `[64,L,256]`。
5. 6 层 Encoder 通过自注意力和前馈网络得到历史 memory。
6. Toy Story 的四位 ID `[204,158,180,0]` 转为标签 `[207,417,695,771,EOS]`。
7. T5 将标签右移为 `[BOS,207,417,695,771]`，送入 6 层 Decoder。
8. Decoder 用因果自注意力读取正确前缀，用交叉注意力读取该用户历史。
9. 输出层得到 `[64,5,1027]` logits，交叉熵计算正确五个 token 的平均损失。
10. 四个 micro-batch 分别 `backward()`，梯度累积后裁剪。
11. AdamW 修改模型参数，scheduler 调整下一 step 的学习率。

### 18.2 验证：模型怎样在不知道答案时接受检验

对用户 0 的 valid 目标 `Mulan (1998)`：

1. 输入只包含训练历史，不把 Mulan 的标签送给 Decoder。
2. 模型从 BOS 开始自回归生成候选路径；Mulan 的真实 token `[25,494,519,771]` 只由评测器保存，用于最后核对排名。
3. Beam Search 保留高分路径，Trie 只允许目录内且用户未见的物品。
4. 四位生成码反查为真实电影，得到 Top-10 排名。
5. 如果 Mulan 出现在 Top-10，Recall@10 为 1；NDCG@10 再根据它的具体名次折扣。否则两项均为 0。

这条链路里，模型真正“学到”的不是一条人工规则，而是 checkpoint 中约 1,128 万个参数。参数共同决定 token 表示、历史如何互相读取、Decoder 如何查询历史，以及每个候选 token 最终获得多高分数。

---

## 19. 初学者最容易混淆的十件事

1. **语义 ID 和模型 embedding 不是同一物。** 前者是固定整数码；后者是 T5 从随机值学出的 256 维参数。
2. **384 维文本向量没有直接输入 T5。** 它已在上一阶段被压缩成四位语义 ID。
3. **attention 不是永久保存的知识表。** 它由当前输入和参数临时计算，换输入就会变化。
4. **`backward()` 不更新参数。** 它只计算并累积 `.grad`；`optimizer.step()` 才更新。
5. **loss 下降不等于推荐指标必然上升。** loss 衡量正确 token 的训练概率，NDCG 衡量未见目标的最终排名。
6. **训练时五个位置可以并行，推理时必须逐步生成。** 原因是训练有真实前缀，推理没有。
7. **Encoder EOS 与 Decoder EOS 用途不同。** 前者标记历史输入结束，后者表示一部物品的四位码生成完毕。
8. **PAD 不代表真实物品。** attention mask 与 `-100` 分别阻止输入 PAD 和标签 PAD 影响结果。
9. **Trie 不提升模型参数。** 它在搜索时排除非法、已见路径；模型训练与约束搜索是两个环节。
10. **生成模型不是自然语言预训练 T5。** 当前结构沿用 T5 Encoder–Decoder，但 1,027 个 token 的含义和全部参数都在本项目数据上从头学习。

理解这十点后，再看训练脚本时，可以始终追踪三样东西：当前张量表示什么、哪些数字是可训练参数、这一行是在算预测、算梯度，还是更新参数。
