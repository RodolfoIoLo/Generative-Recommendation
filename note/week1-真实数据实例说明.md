# 第一周：数据预处理与真实样本说明

> 全部统计数字和片名来自 MovieLens 1M 实测结果（2026-09-17）。标为“示意”的数字不属于实验结果。
> 数据来源：`https://files.grouplens.org/datasets/movielens/ml-1m.zip`
> 配套：《第一周_任务2-3详细任务书.md》（操作手册）· `scripts/preprocess.py` · `scripts/stats.py`

---

## 一句话

| | 任务 2 | 任务 3 |
|---|---|---|
| 比喻 | **备菜** —— 把流水账切成「接龙题」 | **体检** —— 看清这堆数据长什么样 |
| 输出 | train / valid / test + item_text | 统计表 + 长尾图 + 抽样序列 |

下面全部用**真实电影**讲。

---

## 一、任务 2 做了什么——以用户 0 为例

### 1.1 原始文件里是这样的

`ratings.dat` 的原始行（真实格式）：

```
1::1193::5::978300760
1::661::3::978302109
1::914::3::978301968
```

读作「用户 1 在时间 978300760 给电影 1193 打了 5 分」。

**问题**：这是**流水账**，不是**序列**。文件按用户 ID 排，不是按时间排；电影 ID 是原始编号（1~3952 中间有空号）；还混着大量「只被看过一两次」的噪声。

---

### 1.2 5-core：从 3883 部砍到 3416 部

真实数字（实测）：

```
movies.dat 里的电影总数      : 3883
ratings.dat 里出现过的电影数 : 3706      ← 少了 177 部，因为从来没人评分
被评次数 < 5 的电影数        : 290
5-core 后剩余电影数          : 3416      ← 3706 - 290
```

**被砍掉的电影长什么样**（真实例子）：

| 被评次数 | 电影 | 类型 |
|---|---|---|
| 4 次 | `Regret to Inform (1998)` | Documentary |
| 4 次 | `Brother Minister: The Assassination of Malcolm X (1994)` | Documentary |
| 4 次 | `Horror Hotel (a.k.a. The City of the Dead) (1960)` | Horror |
| 4 次 | `Ogre, The (Der Unhold) (1996)` | Drama |

**为什么过滤**：极少交互物品很难获得稳定的行为表示，也会增大评测方差。5-core 是本项目冻结的数据协议，并不意味着被过滤物品本身没有研究价值。

**用户这边呢？一个都没砍。** ml-1m 保证每个用户至少评过 20 部电影，5-core 门槛是 5，所以 6040 个用户全部保留。

> **注意这个细节**：`3883 → 3706` 是“从未出现评分”，`3706 → 3416` 才是“被 5-core 过滤”。公开报告需要区分这两个原因。

---

### 1.3 ID 重映射：Toy Story 从 1 变成 0

原始电影 ID 最大到 **3952**，但只有 **3883** 部，中间有 69 个空号。

重映射后，所有 ID 压成连续的 `0..3415`。真实对应关系：

| 新 ID | 原始 ID | 电影 |
|---|---|---|
| **0** | 1 | `Toy Story (1995)` |
| 1992 | 2018 | `Bug's Life, A (1998)` |
| 1936 | 1962 | `Antz (1998)` |
| 1578 | 1608 | `Mulan (1998)` |
| **46** | 47 | `Pocahontas (1995)` |

**为什么要做**：如果直接拿原始 ID 当索引，embedding 表要开 3953 行，其中 70 行永远用不到 —— 白占显存。而且第 3 周建语义码本、第 5 周算 loss 时，索引对不上会直接崩。

**为什么必须存映射表**：模型输出的永远是「新 ID」。要把 `1578` 翻译回「Mulan (1998)」，只能查 `item_id_map.json`。**这张表丢了，预测结果就永远翻译不回来。**

---

### 1.4 用户 0 的 53 部电影，被切成三份 ⭐ 核心

这是任务 2 真正的样子。用户 0 在真实数据里有 53 条记录，分成两批：

**第一批：2000-12-31，40 部**（一次批量导入，一天之内）

```
Girl, Interrupted (1999)          Back to the Future (1985)       Titanic (1997)
Cinderella (1950)                 Meet Joe Black (1998)           Last Days of Disco, The (1998)
Erin Brockovich (2000)            Christmas Story, A (1983)       To Kill a Mockingbird (1962)
One Flew Over the Cuckoo's Nest   Wallace & Gromit (1996)         Star Wars: Episode IV (1977)
Wizard of Oz, The (1939)          Fargo (1996)                    Run Lola Run (1998)
Rain Man (1988)                   Saving Private Ryan (1998)      Awakenings (1990)
Gigi (1958)                       Sound of Music, The (1965)      Driving Miss Daisy (1989)
Bambi (1942)                      Apollo 13 (1995)                Mary Poppins (1964)
E.T. the Extra-Terrestrial (1982) My Fair Lady (1964)             Ben-Hur (1959)
Big (1988)                        Sixth Sense, The (1999)         Dead Poets Society (1989)
James and the Giant Peach (1996)  Ferris Bueller's Day Off (1986) Secret Garden, The (1993)
Toy Story 2 (1999)                Airplane! (1980)                Pleasantville (1998)
Dumbo (1941)                      Princess Bride, The (1987)      Snow White (1937)
Miracle on 34th Street (1947)
```

**第二批：2001-01-06，13 部**（一周后，这才是关键）

| 拆分 | 电影 | 类型标签（真实） |
|---|---|---|
| train | `Ponette (1996)` | Drama |
| train | `Schindler's List (1993)` | Drama\|War |
| train | `Beauty and the Beast (1991)` | Animation\|Children's\|Musical |
| train | `Tarzan (1999)` | Animation\|Children's |
| train | `Close Shave, A (1995)` | Animation\|Comedy |
| train | `Aladdin (1992)` | Animation\|Children's\|Comedy\|Musical |
| train | `Toy Story (1995)` | Animation\|Children's\|Comedy |
| train | `Bug's Life, A (1998)` | Animation\|Children's\|Comedy |
| train | `Antz (1998)` | Animation\|Children's |
| train | `Hunchback of Notre Dame, The (1996)` | Animation\|Children's\|Musical |
| train | `Hercules (1997)` | Adventure\|Animation\|Children's\|Comedy\|Musical |
| **valid** | **`Mulan (1998)`** | Animation\|Children's |
| **test** | **`Pocahontas (1995)`** | Animation\|Children's\|Musical\|Romance |

这段序列呈现出清晰规律：该用户连续记录了 **11 部动画片**，随后：

- 验证集要他预测：下一部是什么？→ 答案是 **Mulan**
- 测试集要他预测：再下一部？→ 答案是 **Pocahontas**

两部都是迪士尼动画。**这就是「序列推荐」的全部内容**：给他一串历史，猜下一个。

---

### 1.5 最终产出

```json
// train.jsonl —— 51 部，训练用
{"user": 0, "items": [2757, 861, 1068, 1444, 1978, ..., 1992, 1314]}

// valid.jsonl —— 用前 51 部预测第 52 部
{"user": 0, "history": [2757, 861, ..., 1992, 1314], "target": 1578}
//                                                              ↑ Mulan

// test.jsonl —— 用前 52 部预测第 53 部
{"user": 0, "history": [2757, 861, ..., 1992, 1314, 1578], "target": 46}
//                                                                   ↑ Pocahontas
```

**任务 2 干的事，就是把原始流水账变成这三个文件。** 模型的工作就是「接龙」。

---

## 二、任务 3 做了什么 —— 真实统计

### 2.1 统计表（真实数字）

| 指标 | 值 |
|---|---|
| 用户数 | **6040** |
| 物品数 | **3416** |
| 交互总数 | **999,611** |
| 稀疏度 | **0.9516** |
| 每用户平均序列长度 | **165.5** |
| 中位序列长度 | **96** |
| P90 序列长度 | **400** |
| 最长 / 最短序列 | **2277 / 18** |
| 最热物品被看次数 | **3428** |
| 物品被看次数中位数 | **146** |
| 基尼系数 | **0.6034** |
| 时间跨度 | 2000-04-25 ~ 2003-02-28 |

**稀疏度 0.9516 是什么意思**：`1 - 999611 / (6040 × 3416) = 0.9516`。也就是说，用户-物品矩阵里 **95% 的格子是空的**。这就是推荐任务难的根本原因。

---

### 2.2 长尾：真实的最热和最冷

**最热的 5 部**：

| 被看次数 | 电影 |
|---|---|
| 3428 | `American Beauty (1999)` |
| 2991 | `Star Wars: Episode IV - A New Hope (1977)` |
| 2990 | `Star Wars: Episode V - The Empire Strikes Back (1980)` |
| 2883 | `Star Wars: Episode VI - Return of the Jedi (1983)` |
| 2672 | `Jurassic Park (1993)` |

**最冷的 5 部**（5-core 之后仍然留下的）：

| 被看次数 | 电影 | 类型 |
|---|---|---|
| 5 | `Machine, The (1994)` | — |
| 5 | `Delta of Venus (1994)` | Drama |
| 5 | `Boys Life 2 (1997)` | — |
| 5 | `Return with Honor (1998)` | Documentary |
| 5 | `Two or Three Things I Know About Her (1966)` | Drama |

**同一个类型内部的差距更直观**：

| 类型 | 最热 | 最冷 | 差距 |
|---|---|---|---|
| Horror | `Ghostbusters (1984)` — **2181 次** | `City of the Living Dead (1980)` — **5 次** | **436 倍** |
| Documentary | `Roger & Me (1989)` — **798 次** | `Return with Honor (1998)` — **5 次** | **160 倍** |

---

### 2.3 ⚠️ 一个必须诚实面对的发现

**5-core 过滤本身就把最冷的那批物品砍掉了。**

所以剩下的长尾看起来**没有想象中极端**：

- 交互数 ≤ 10 的物品只占 **5.4%**（因为 < 5 的已经被砍了）
- 基尼系数 **0.6034** —— 属于中等偏上，不算极端
- 最热 1% 的物品吃掉 **8.2%** 的交互，最热 10% 吃掉 **42.3%**

**这意味着什么**：写报告时**不要夸大**说「长尾极其严重」。但也别因此觉得课题没意义 —— 真正的论证点应该换成：

> **即使已经过滤到 5-core，仍有 26.8% 的物品只有 50 次以内交互**（3416 部里有 917 部）。仅依赖行为序列的基线并未直接利用这些物品的片名与类型；而类型标签本身已经提供了「恐怖片」「纪录片」等内容信息。

这个说法比「长尾严重」更准确，也更站得住脚。

---

### 2.4 抽样看序列发现了什么

**发现一：用户序列存在局部语义连贯性** ✅

用户 0 连续交互的 11 部动画片提供了一个直观例子：行为序列并非完全随机，局部历史能够包含类型与主题偏好。这为使用语义标识符提供了动机，但不能单凭一条用户序列证明语义 ID 一定有效；其前缀一致性仍需在全体物品上统计验证。

**发现二：数据有批量导入痕迹** ⚠️

用户 0 在 **2000-12-31 一天之内**录入 40 部电影。这些时间戳更可能记录集中评分或批量补录，而不是 40 次真实观影发生的时间。因此 MovieLens 时间戳应解释为交互记录时间，不能直接等同于观看时间。

**影响**：leave-one-out 按时间划分时，最后两条（Mulan、Pocahontas）恰好都在 2001-01-06 那批里，时间顺序并无异常。但如果某用户最后一批是集中补录的，划分结果未必等同于真实观看顺序。这是解释实验结论时需要保留的数据局限。

---

## 三、同一批数据如何进入后续模型

### 任务 4：判别式基线在预测什么

回到用户 0。模型的输入输出是：

```
输入（51 部）：... Schindler's List, Beauty and the Beast, Tarzan, Aladdin,
               Toy Story, Bug's Life, Antz, Hunchback, Hercules
                    ↓
              模型（黑盒）
                    ↓
输出：从 3416 部电影里，挑出最可能的下一部 → Mulan (1998)
```

对 SASRec、GRU4Rec 等判别式基线而言，本质是对 3,416 部目录电影打分并排序。第五至六周的生成模型则逐 token 生成四位语义 ID，两类模型最终都输出 Top-K 电影列表。

### 任务 5：预测对了怎么算分

模型不会只给一个答案，它给一个**排序**。假设给用户 0 的 Top-3 是：

| 情况 | Top-3 排序 | Recall@3 | NDCG@3 |
|---|---|---|---|
| 理想 | `Mulan, Pocahontas, Tarzan` | 1 ✅ | **1.000**（排第一，不打折） |
| 次优 | `Tarzan, Mulan, Pocahontas` | 1 ✅ | **0.631**（排第二，打 1/log₂3 折） |
| 失败 | `Tarzan, Hercules, Antz` | 0 ❌ | **0.000** |

**两种情况下 Recall 都是 1，但 NDCG 差很多** —— 这就是为什么推荐系统更看重 NDCG：**命中还不够，得排在前面**。（这些数字是用脚本算过的，不是估的。）

### 任务 6：SASRec 和 TIGER 的差别

拿用户 0 举例，两种范式的做法完全不同：

| | **SASRec**（判别式基线） | **本项目生成模型**（TIGER-style） |
|---|---|---|
| 每部电影是什么 | 一个目录编号，`Mulan = 1578` | 四位语义 ID，`Mulan = [22,235,4,0]` |
| 预测方式 | 给 3,416 部电影打分 | 生成四个位置 token 与 EOS |
| 长尾物品的信息 | 主要从有限行为中学习独立 item embedding | 固定语义 ID 来自文本，可与其他物品共享 token 和模型参数 |

最后一行是本项目研究生成式推荐的主要动机。严格冷启动收益仍需包含未见目标的专门划分才能直接验证；当前划分的 strict-cold 目标数为 0。

### 阶段二（第 3–4 周）：构建语义标识符

用任务 2 导出的 `item_text.json`（真实内容）：

```
[0]    Toy Story (1995)     | Animation|Children's|Comedy
[1992] Bug's Life, A (1998) | Animation|Children's|Comedy
[1936] Antz (1998)          | Animation|Children's
[1578] Mulan (1998)         | Animation|Children's
[46]   Pocahontas (1995)    | Animation|Children's|Musical|Romance
```

**流程**：文本 → 预训练编码器转成向量 → RQ-VAE 量化成分层码。

**最终冻结的真实四位语义 ID 如下**：

```
Toy Story (1995)       → [204, 158, 180, 0]
Bug's Life, A (1998)   → [250, 207, 245, 0]
Antz (1998)            → [250, 157, 128, 0]
Mulan (1998)           → [22, 235, 4, 0]
Pocahontas (1995)      → [225, 40, 11, 0]
```

这些个例也说明：语义相近不等于必然共享前缀。项目通过全体物品统计验证了反方向的命题——**共享前缀的物品整体上比随机物品对更相似**。该结论是群体规律，不应被解释为人工类别标签。

**这就是「把物品变成词」的含义** —— 电影不再是孤立的编号，而是「有语义归属的词」。

### 阶段三（第 5–6 周）：生成式模型

```
输入：用户 0 的历史 → 每部电影转成四个位置 token，展平并追加 EOS
                                    ↓
                     随机初始化的 T5 式 Encoder–Decoder
                                    ↓
输出：生成 Mulan 对应的 token `[25,494,519,771]` 与 EOS
                                    ↓
                     还原四位 ID `[22,235,4,0]`
                                    ↓
                         查表得到 `Mulan (1998)`
```

**约束解码**使用动态 Trie，保证 token 路径对应目录内真实电影，并排除该用户已经交互过的电影。

---

## 四、总结

| 阶段 | 用真实数据说 | 产出 |
|---|---|---|
| **任务 2** | 用户 0 的 53 部电影 → train 51 / valid 1 / test 1 | `train.jsonl` 等 7 个文件 |
| **任务 3** | 6040 用户 / 3416 电影 / 999611 交互；最热 3428 次 vs 最冷 5 次 | 统计表 + 长尾图 + 抽样序列 |
| **任务 4–6** | 模型要预测 `Mulan`；排第一得 1.0 分，排第二得 0.63 分 | 理解 + 对比笔记 |
| **阶段二** | `Toy Story` → `[204,158,180,0]`（实测） | 语义标识符 |
| **阶段三** | 生成 token → 翻译回 `Mulan (1998)` | 生成式模型 |

任务 2 把 53 部电影切成“用前 51 部预测第 52 部”的问题，任务 3 描述数据规模与长尾分布；后续模型在同一划分下学习把 `Mulan`、`Pocahontas` 等真实目标尽量排在推荐列表前部。
