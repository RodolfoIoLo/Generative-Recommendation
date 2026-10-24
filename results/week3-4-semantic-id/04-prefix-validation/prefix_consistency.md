# 前缀语义一致性验证

embedding: `item_emb.npy`|语义 ID: `semantic_ids_rqvae.json`|bootstrap 1000 次

| 前缀深度 | 同前缀物品对平均余弦 | 随机配对平均余弦 | 差值 | 差值的 pair-bootstrap 95% CI | 组数 | 对数 |
|---|---|---|---|---|---|---|
| 前 1 位 | 0.5289 | 0.2832 | **+0.2456** | [0.2436, 0.2476] | 256 | 29251(bootstrap 已封顶) |
| 前 2 位 | 0.7116 | 0.2832 | **+0.4284** | [0.4079, 0.4498] | 3234 | 195 |
| 前 3 位 | 0.8089 | 0.2832 | **+0.5257** | [0.4101, 0.6314] | 3411 | 5 |

三种深度下，同前缀物品对均稳定地比随机物品对更相似；差值的 95% 区间均完全大于 0。这里的区间是物品对层面的描述性 pair-bootstrap，不作 MovieLens 之外的总体显著性或因果推断。

## 人工检查

从 `prefix_samples.csv` 按前两位前缀抽查。下列 10 组在题材、类型或系列关系上具有清晰的一致性：

| 前缀 | 抽样物品 | 人工判断 |
|---|---|---|
| `(83,181)` | French Twist / French Kiss / Forget Paris | 同为 1995 年喜剧爱情片 |
| `(154,40)` | The Paper / Reality Bites / With Honors | 同为 1994 年喜剧剧情片 |
| `(61,76)` | No Looking Back / Can't Hardly Wait / Living Out Loud | 以喜剧、剧情和爱情为共同主题 |
| `(66,252)` | Friday the 13th 及 Part 2 / Part 3 | 同一恐怖电影系列 |
| `(169,250)` | House / Class of Nuke 'Em High / Night of the Creeps | 1986 年喜剧恐怖片 |
| `(82,213)` | Superman II / III / IV | 同一超级英雄系列，类型完全一致 |
| `(124,103)` | Teenage Mutant Ninja Turtles I / II / III | 同一忍者神龟系列 |
| `(66,235)` | Slumber Party Massacre I / II / III | 同一恐怖电影系列 |
| `(43,40)` | The Juror / Diabolique | 同为 1996 年剧情惊悚片 |
| `(28,223)` | NeverEnding Story II / III | 同一奇幻冒险系列 |

同时保留两组反例，不把共享前缀解释成严格的人工类别标签：

- `(177,88)`：*The Exorcist III*（恐怖）与 *Arachnophobia*（动作/喜剧/科幻/惊悚）共享前缀，但类型标签无交集；二者余弦为 0.6028。
- `(83,161)`：*Living in Oblivion*（电影制作题材）与 *Empire Records*（唱片店青春题材）共享前缀，但叙事主题明显不同；二者余弦为 0.5768。

## 候选方案对照与冻结结论

对四层纯 RQ-VAE 候选采用同一随机池和 bootstrap 设置复验，其前 1/2/3 位差值分别为 `+0.2439`、`+0.4188`、`+0.5419`，与正式三层方案接近。考虑到三层方案有正式 checkpoint、可从 checkpoint 重算并与 JSON 完全一致，且符合“残差量化前缀 + 确定性消歧码”的既定协议，最终冻结 `L=3, K=256` 的 RQ-VAE 前缀，并追加一位消歧码。四层纯语义方案保留为后续消融实验候选。
