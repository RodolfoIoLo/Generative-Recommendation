# 🏁 正式测试结果

Text、Fixed、SASRec、GRU4Rec各三个种子，12项test全部完成，每项覆盖6,040名用户。候选和训练配置在测试前冻结。

先读[最终报告](main_results.md)：四项总体指标、各种子、长尾／少样本分组，以及Fixed相对Text和GRU4Rec的用户配对区间。

| 位置 | 内容 |
|---|---|
| `generative/{text,fixed}/seed{42,43,44}/` | 生成模型指标与固定20名用户样例 |
| `baselines/{sasrec,gru4rec}/seed{42,43,44}/` | 同h20基线指标与固定20名用户样例 |
| [summaries/test_results.csv](summaries/test_results.csv) | 四种模型×五个分组的四项指标，均值与总体标准差 |
| `summaries/summary.json`、`baseline_h20_summary.json` | 生成模型／基线的三种子汇总 |
| `summaries/paired_bootstrap.json` | Fixed−Text各分组、各指标的用户配对95%区间 |
| `summaries/gru4rec_comparison.json` | Fixed−GRU4Rec总体NDCG@10的配对95%区间 |
| [figures/test_comparison.png](figures/test_comparison.png) | 总体与长尾对比；误差线为种子标准差 |

完整6040用户预测保留在实验工作区`work/week7/predictions/`，汇总脚本会逐项校验用户、目标、历史屏蔽、Top-10唯一性和指标。生成模型无非法ID、已见物品或不足10项的列表。严格冷启动test为0人，指标为`null`／N/A。
