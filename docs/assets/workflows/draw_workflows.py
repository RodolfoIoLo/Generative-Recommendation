"""Draw the five file-level learning maps and the README overview (CPU only)."""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent
W = 3600
COLORS = {"ink": "#213447", "muted": "#526577", "paper": "#f5f7fa",
          "line": "#c3cfd9", "input": "#fff1da", "code": "#e7efff", "output": "#e0f3e9"}

# Each row is one step on the main trunk. Its three branches identify the
# actual input, executable and output; paths are relative to the run root.
MAPS = [
    ("week1", "WEEK 1", "把观看记录整理成可学习的数据", "#b77d27",
     "起点：MovieLens 1M原始记录　→　终点：固定划分、电影文本与数据分布",
     [
         ("01  读懂一条交互", [
             ("输入 · 原始数据", ["ml-1m/raw/ratings.dat", "用户、电影、评分、时间戳", "ml-1m/raw/movies.dat", "电影标题与类型；按原始电影ID连接"]),
             ("程序 · 预处理入口", ["scripts/week1_preprocess.py", "读取原始记录并检查必要字段", "本项目把观看记录作为隐式反馈", "评分值不作为推荐目标"]),
             ("得到 · 可排序的事件", ["用户—电影—时间的交互序列", "文本与交互使用相同电影映射", "原始文件在数据工作区", "不纳入Git"]),
         ], "数据根目录由 --data-root 指定；原始数据与公开结果分开存放。"),
         ("02  清洗与连续编号", [
             ("输入 · 交互与电影信息", ["上一阶段的原始事件", "电影必须有对应文本信息", "迭代5-core过滤", "用户与物品都满足最少交互要求"]),
             ("程序 · 映射与序列", ["scripts/week1_preprocess.py", "过滤后建立连续内部ID", "用户内部ID与电影内部ID各自独立", "按时间排列每人的观看记录"]),
             ("得到 · 电影字典", ["processed/user_id_map.json", "processed/item_id_map.json", "processed/item_text.json", "电影内部ID → 标题与类型文本"]),
         ], "清洗后：6,040名用户、3,416部电影。后续数组第i行始终对应电影内部ID i。"),
         ("03  划分训练、验证、测试", [
             ("输入 · 按时间排序的用户序列", ["每人保留自己的观看顺序", "倒数第二次交互作为valid目标", "最后一次交互作为test目标", "更早的交互只用于train"]),
             ("程序 · 划分与检查", ["scripts/week1_preprocess.py", "scripts/week1_verify_split.py", "校验人数、顺序与目标连接关系", "train / valid / test各6,040行"]),
             ("得到 · 两类数据接口", ["processed/{train,valid,test}.jsonl", "train为完整训练序列", "valid/test为历史＋下一部电影", "processed/ml-1m.inter供RecBole读取"]),
         ], "train共987,531次交互。后续生成训练在线展开为981,491个前缀—目标样本。"),
         ("04  看清数据分布", [
             ("输入 · 清洗后的数据", ["processed/下的划分与电影文本", "序列长度、电影出现次数", "全体清洗交互用于描述性统计", "这不是Week 7的train-only频次"]),
             ("程序 · 数据分析", ["scripts/week1_stats.py", "scripts/week1_export_item_frequency.py", "计算规模与分布", "生成统计JSON、Markdown和直方图"]),
             ("得到 · 可阅读证据", ["processed/stats/stats.{json,md}", "processed/stats/figs/*.png", "results/week1-data-stats/item_freq.json", "预处理数量记录：preprocess_log.json"]),
         ], "统计报告与图从数据工作区归档到 results/week1-data-stats/，内容保持一致。"),
         ("05  交给后续两条路线", [
             ("分支A · 序列推荐", ["ml-1m.inter → Week 2", "用已有方法建立效果参照", "train/valid/test决定学习与评测", "ID映射确保能查回同一部电影"]),
             ("分支B · 电影内容", ["item_text.json → Week 3–4", "用文本编码器得到电影向量", "同一电影顺序贯穿量化和生成训练", "避免文本行与交互ID错位"]),
             ("仓库 · 本周归档", ["results/week1-data-stats/", "stats.md / stats.json", "preprocess_log.json / item_freq.json", "figs/：序列长度、频次、长尾图"]),
         ], "终点检查：数据身份固定、划分可复验、电影文本与内部ID对齐；原始数据留在数据工作区。"),
     ]),
    ("week2", "WEEK 2", "先让已有推荐方法跑通", "#476db3",
     "起点：固定交互划分　→　终点：SASRec与GRU4Rec三种子基线及原始证据",
     [
         ("01  连接同一份数据", [
             ("输入 · Week 1", ["processed/ml-1m.inter", "6,040名用户、3,416部电影", "原始时间信息用于顺序划分", "recbole/ml-1m链接到processed/"]),
             ("配置 · 实验条件", ["config/sasrec_ml1m.yaml", "RecBole读取电影交互", "旧基线历史上限200", "按验证指标选择checkpoint"]),
             ("得到 · 学习任务", ["历史电影 → 下一部电影", "每部电影是一个普通物品ID", "此时不使用文本向量或语义ID", "两种方法共享数据处理口径"]),
         ], "基线是比较参照，不是生成模型的预训练权重。"),
         ("02  SASRec学习序列", [
             ("输入 · 历史物品ID", ["训练序列的历史前缀", "电影ID映射为模型可学习向量", "位置告诉模型观看顺序", "用户未来的目标不进入训练输入"]),
             ("程序 · 自注意力模型", ["scripts/week2_train_baseline.py", "--model SASRec", "RecBole管理训练与验证", "seed 42 / 43 / 44分别训练"]),
             ("得到 · 分数与权重", ["为目录电影打分并排序", "saved/SASRec-*.pth", "log/SASRec/中的原始运行日志", "log_tensorboard/中的事件文件"]),
         ], "三个随机种子衡量训练波动；checkpoint内配置和日志确认seed，不能只看文件名猜测。"),
         ("03  GRU4Rec学习序列", [
             ("输入 · 相同观看序列", ["与SASRec相同的数据来源", "按先后顺序读取历史电影", "不读取电影简介", "不使用生成式语义ID"]),
             ("程序 · 循环神经网络", ["scripts/week2_train_baseline.py", "--model GRU4Rec", "GRU把历史累积为隐藏状态", "seed 42 / 43 / 44分别训练"]),
             ("得到 · 第二种参照", ["saved/GRU4Rec-*.pth", "log/GRU4Rec/中的原始日志", "各自的best valid与test指标", "目录电影按预测分数排序"]),
         ], "两种结构分别学习，权重不共享；它们提供不同的序列建模参照。"),
         ("04  复评与保留原始记录", [
             ("输入 · 已训练checkpoint", ["模型权重及checkpoint配置", "对应训练日志中的模型与seed", "验证/测试目标保持固定", "无需为了补指标重新训练"]),
             ("程序 · 只加载并评测", ["scripts/week2_evaluate_checkpoint.py", "加载已选出的模型", "补齐缺失评测输出", "与原始日志核对对应关系"]),
             ("得到 · 仓库证据", ["results/week2-sasrec-baseline/", "training-logs/{sasrec,gru4rec}/", "evaluation-logs/sasrec/", "baseline_table.md汇总三个种子"]),
         ], "原始RecBole评测与后续完整历史屏蔽口径不同，旧表保留为历史结果，不改写成新分数。"),
         ("05  与后续实验衔接", [
             ("Week 5–6 · 对齐历史屏蔽", ["week5_6_evaluate_baselines.py", "加载本周checkpoint", "同时输出完整历史屏蔽与旧口径", "校验旧日志，避免数据/ID错位"]),
             ("Week 7 · 重训h20参照", ["week7_baseline.py", "历史上限改为20，与生成模型一致", "按屏蔽完整历史的valid重新选模", "三种子训练后单独做正式test"]),
             ("终点 · 会读比较表", ["Week 2表：原始口径", "Week 5–6表：旧h200模型复评", "Week 7表：最终h20受控比较", "不能跨表直接归因模型结构优劣"]),
         ], "终点检查：两种基线×三个种子完成；日志、配置、checkpoint与指标能够对应。"),
     ]),
    ("week3-4", "WEEK 3–4", "从电影简介长出一棵编号树", "#8754ac",
     "起点：电影文本　→　384维向量　→　三层残差量化　→　唯一四段ID与六文件接口",
     [
         ("01  把文本变成数字", [
             ("输入 · 标题与类型", ["processed/item_text.json", "电影内部ID连续为0…3415", "共享编码器：all-MiniLM-L6-v2", "本项目使用现成文本模型权重"]),
             ("程序 · 文本编码", ["scripts/week3_4_encode_items.py", "编码标题和类型文本", "按内部ID顺序保存数组", "对每行做L2归一化"]),
             ("得到 · 电影文本向量", ["item_emb.npy：3416×384", "第i行对应电影i", "行范数约1，全部数值有限", "向量缓存留在实验工作区"]),
         ], "文本向量描述语义相似性，不是推荐模型的词表，也不是生成模型的训练权重。"),
         ("02  两种残差量化路线", [
             ("输入 · 单位电影向量", ["item_emb.npy", "RQ-KMeans：直接量化384维输入", "RQ-VAE：先压缩到32维潜空间", "三层，每层256个代表向量"]),
             ("程序 · 量化与坍缩修复", ["scripts/week3_4_build_semantic_ids.py", "连续AE预训练200轮", "逐层K-Means初始化残差码本", "重建＋码本＋承诺损失联合训练"]),
             ("得到 · 三位码与权重", ["01-quantization/identifiers/", "semantic_ids_{kmeans,rqvae}.json", "saved/rq_kmeans_ml1m.npz", "saved/rqvae_ml1m.pt"]),
         ], "首次随机初始化坍缩为1个ID；修复后正式RQ-VAE重建R²=0.623914，三层使用256/253/253个码字。"),
         ("03  给相同编号追加消歧位", [
             ("输入 · 三层码的选择编号", ["第0层选中心 → c0", "扣除该中心后第1层选中心 → c1", "再对剩余误差选第2层 → c2", "相同(c0,c1,c2)可能对应多部电影"]),
             ("程序 · 确定性去重", ["week3_4_build_semantic_ids.py", "按共享三位码分组", "组内按物品顺序追加序号", "形成(c0,c1,c2,消歧序号)"]),
             ("得到 · 唯一语义ID", ["每部电影一个四段编号", "正式RQ-VAE原三位有5个重复ID", "最大冲突组2部电影", "追加后3416个ID全部唯一"]),
         ], "第四段是消歧序号，不是语义码本；唯一性保证完整ID可以无歧义地查回电影。"),
         ("04  看码本是否真的在工作", [
             ("输入 · ID与电影向量", ["两种方法的完整语义ID", "不把消歧位计为语义码本", "统计码字使用量与分配分布", "比较共享前缀的电影相似度"]),
             ("程序 · 两类质量检查", ["week3_4_analyze_codebook.py", "week3_4_check_prefix.py", "利用率／困惑度／冲突率", "共享前缀余弦相似度与人工样例"]),
             ("得到 · 可检查的解释", ["02-codebook-analysis/codebook_stats.json", "02-codebook-analysis/figures/方法/", "04-prefix-validation/prefix_consistency.md", "04-prefix-validation/prefix_samples.csv"]),
         ], "高利用率说明没有大面积闲置码字；重建质量和前缀语义不能替代最终推荐效果。"),
         ("05  做网格并保留正式版本", [
             ("输入 · 不同码本容量", ["主网格：L∈{2,3}，K∈{64,256}", "两种方法，共8组", "额外：L=4、K=256纯RQ-VAE", "额外组关闭人工消歧"]),
             ("程序 · 受控改变容量", ["week3_4_build_semantic_ids.py", "分别设置 --n_levels / --codebook_size", "相同输入与seed，分别保存候选", "正式接口沿用已保存可复验主权重"]),
             ("得到 · 横向对照", ["03-grid-search/grid_search.md", "candidates/{kmeans,rqvae}/", "logs/{kmeans,rqvae}/", "四层纯量化本次也达到3416个唯一ID"]),
         ], "网格与正式运行可能存在GPU数值差异；最终ID以正式checkpoint及其冻结哈希为准。"),
         ("06  把全部序列换成语义ID", [
             ("输入 · ID映射＋原始划分", ["semantic_ids_rqvae.json", "processed/{train,valid,test}.jsonl", "同一内部电影ID查到同一四段编号", "保留用户和目标关系"]),
             ("程序 · 冻结模型接口", ["scripts/week3_4_export_semantic.py", "movie ID → 四段语义ID", "构建分位置Token词表", "记录输入、量化权重与划分SHA-256"]),
             ("得到 · 六份冻结文件", ["semantic_ids.json / token_vocab.json", "train.semantic.jsonl", "valid.semantic.jsonl / test.semantic.jsonl", "meta.json：协议与来源身份"]),
         ], "终点：results/week3-4-semantic-id/05-frozen-interface/公开六文件；工作区exported/供训练读取。"),
     ]),
    ("week5-6", "WEEK 5–6", "让模型学会逐位生成下一部电影", "#297f94",
     "起点：六份冻结语义接口　→　终点：三种子生成模型、合法推荐列表与首版结果",
     [
         ("01  确认输入与输出长什么样", [
             ("输入 · Week 3–4接口", ["exported/下的六份冻结文件", "电影四段ID → 四个位置Token", "词表4×256+3=1027", "PAD=0，BOS=1，EOS=2"]),
             ("程序 · 任务审计", ["scripts/week5_6_common.py", "scripts/week5_6_train.py --audit_only", "检查Token可逆、用户/电影数量", "确认981491个训练前缀"]),
             ("得到 · 训练任务", ["历史最多20部电影 → 最多81 Token", "目标：下一部的4个Token＋EOS", "01-pipeline-validation/audit/", "审计日志与静态检查记录"]),
         ], "四段ID不直接作为浮点输入；它们先转换成位置独立的离散Token。"),
         ("02  看清模型内部在学什么", [
             ("输入 · 历史与真实目标", ["前缀样本在线展开", "Encoder读取历史Token与位置关系", "Decoder读取BOS＋真实目标前缀", "目标为四个ID Token和EOS"]),
             ("程序 · 自回归训练", ["scripts/week5_6_train.py", "T5式6层Encoder＋6层Decoder", "随机初始化11281664个参数", "交叉熵 → 反向传播 → AdamW更新"]),
             ("得到 · 可恢复的训练状态", ["saved/generative/运行名/{best,last}/", "model.safetensors与模型配置", "trainer_state.pt保存优化器等状态", "冻结语义ID不参与推荐梯度更新"]),
         ], "teacher forcing：训练给真实前缀，推理用已生成前缀；T5 Token embedding和RQ码本是不同参数。"),
         ("03  先验证管线，再选学习率", [
             ("输入 · 小规模诊断任务", ["冒烟20步再恢复至30步", "256训练样本过拟合", "lr=3e-4与1e-3各跑5000步", "预实验只用valid选参数"]),
             ("程序 · 同一训练入口", ["week5_6_train.py的限制参数", "保存/加载模型和优化器", "检查loss可下降、恢复链路连续", "week5_6_report.py --pilot_only"]),
             ("得到 · 训练前证据", ["01-pipeline-validation/{smoke,overfit}/", "02-learning-rate-selection/lr3e-4/", "02-learning-rate-selection/lr1e-3/", "pilot_summary.md：选择3e-4"]),
         ], "256样本loss降至0.001613验证可学习性；该检查不解释为正式推荐能力。"),
         ("04  三个种子完整训练", [
             ("输入 · 固定正式协议", ["config/generative_ml1m.yaml", "seed42、43、44独立训练", "有效batch=64×4=256", "30k预算，线性warmup与衰减"]),
             ("程序 · 训练与valid选模", ["scripts/week5_6_train.py", "每5000步评估valid NDCG@10", "更高valid更新best", "last用于中断恢复"]),
             ("得到 · 按种子归档", ["03-formal-training/seed{42,43,44}/", "train_seed*.log：原始输出", "t5_sid_seed*_history.jsonl：轨迹", "*_best_valid_metrics.json：选模记录"]),
         ], "训练过程和选模访问train与valid，test不参与参数或候选选择。"),
         ("05  生成候选并查回电影", [
             ("输入 · best＋验证/测试历史", ["历史由相同冻结接口读取", "Beam Search保留20条前缀", "逐位生成4段编号与EOS", "完整历史决定哪些电影已看过"]),
             ("程序 · 两种解码对照", ["scripts/week5_6_evaluate.py", "constrained：动态Trie约束合法ID", "unconstrained：不加Trie约束", "查ID、去重复、计算Top-5/10指标"]),
             ("得到 · 预测与指标", ["04-generative-evaluation/valid/解码方式/", "04-generative-evaluation/test/解码方式/", "12份指标JSON，另有完整预测JSONL", "约束生成无非法／已见／短列表"]),
         ], "全目录候选、不做负采样；Beam20近似搜索不等同于逐一精确计算全部电影概率。"),
         ("06  用同口径解释结果", [
             ("输入 · 生成结果＋旧基线权重", ["Week 2的SASRec／GRU4Rec", "保持同一用户与目标", "完整历史屏蔽后重新评测", "同时复现原始日志指标"]),
             ("程序 · 复评与汇总", ["week5_6_evaluate_baselines.py", "week5_6_report.py", "按train频次分组、三种子统计", "抽取固定20名用户的解码样例"]),
             ("得到 · 首版结果", ["05-baseline-evaluation/{sasrec,gru4rec}/", "06-reports/main_results.md", "06-reports/decode_samples.md", "生成式test NDCG@10=0.1180±0.0092"]),
         ], "终点：完整生成推荐链路跑通；首版效果落后基线，成为Week 7受控改进的起点。"),
     ]),
    ("week7", "WEEK 7", "把观看行为融入编号，再做公平比较", "#27846b",
     "起点：文本向量与train行为　→　终点：四组消融、12项正式test与可追溯结论",
     [
         ("01  先确认瓶颈与输入身份", [
             ("输入 · 旧实验与冻结接口", ["旧三种子训练轨迹", "旧seed42的30k best权重", "同一批512名valid用户", "exported/、item_emb.npy"]),
             ("程序 · 诊断与初始化", ["scripts/week7_run.py init / curves", "week7_evaluate.py：Beam与exact", "week7_report.py diagnostics", "只比较搜索，不用test调参"]),
             ("得到 · 搜索与训练证据", ["01-diagnostics/environment/", "01-diagnostics/search/{metrics,samples}/", "search/search_diagnosis.md", "training-curves/最终含11次运行114点"]),
         ], "Beam20与exact Top-10重合率90.61%；exact是概率排序参考，并非NDCG上界。"),
         ("02  仅从train提取协同关系", [
             ("输入 · 冻结train序列＋文本", ["train.semantic.jsonl还原电影ID", "用户—电影二值矩阵6040×3416", "同一用户重复观看只占一个矩阵项", "train交互次数另外统计权重频次"]),
             ("程序 · SVD与四种融合", ["week7_prepare.py features", "矩阵转置后TruncatedSVD得到64维", "文本384维与协同64维分别归一化", "[文本, 权重×协同]再整体归一化"]),
             ("得到 · 四组448维输入", ["work/week7/collab.npy", "work/week7/{text,fixed,fixed_match,adaptive}.npy", "02-embeddings/{meta,weights,neighbors}.json", "Text末64维为0；Fixed权重1"]),
         ], "Fixed-Match匹配Adaptive平均协同平方范数占比；Adaptive权重由train频次决定。valid/test不进入SVD。"),
         ("03  为四组重新训练量化器", [
             ("输入 · 相同维度的四组表示", ["每组3416×448，归一化", "config/week7/rqvae.yaml", "相同32维潜空间、3×256码本", "量化随机种子均为42"]),
             ("程序 · 量化与冻结导出", ["week7_prepare.py tokenize --variant", "调用week3_4_build_semantic_ids.py", "AE预训练→残差初始化→联合训练", "重新映射同一用户/目标划分"]),
             ("得到 · 四套ID与证据", ["03-semantic-ids/方案/identifiers/", "quality/、training/、figures/", "exported/week7/方案/六份接口", "saved/week7/quantizers/方案.pt"]),
         ], "四组消歧后均3416个唯一ID，三层各至少231个码字；R²目标不同，不用于排名推荐效果。"),
         ("04  先筛选，再冻结候选", [
             ("输入 · 四套语义ID", ["config/week7/generative_方案.yaml", "相同T5结构、60k预算、有效batch256", "peak lr=3e-4，余弦衰减", "最近20部电影，Beam20"]),
             ("程序 · 四组seed42", ["week7_train.py训练并按valid选best", "week7_report.py select", "比较valid NDCG@10", "禁止test参与候选选择"]),
             ("得到 · 一次冻结选择", ["04-screening/training-logs/", "04-screening/ablation.json", "selection.json：选择Fixed", "Text .17761 → Fixed .19352（valid）"]),
         ], "seed42筛选同时计入正式Text/Fixed种子，避免把同一个运行重复算成两次实验。"),
         ("05  补齐训练与同h20基线", [
             ("输入 · 已冻结候选", ["Text和Fixed各补seed43、44", "SASRec／GRU4Rec各seed42、43、44", "基线历史上限同样20", "按完整历史屏蔽后的valid选模"]),
             ("程序 · 正式训练", ["week7_train.py：生成模型", "week7_baseline.py：判别式基线", "checkpoint安全写入与恢复", "模型权重与输入身份一致性检查"]),
             ("得到 · 模型与训练证据", ["05-formal-training/generative/方案/seed/", "05-formal-training/baselines/模型/seed/", "saved/week7/generative/运行名/{best,last}/", "saved/week7/baselines/运行名/"]),
         ], "共8次生成模型训练（四组seed42＋Text/Fixed各补两个种子），6次基线训练；固定预算不等于充分收敛。"),
         ("06  全量test与重新核算", [
             ("输入 · 12个冻结best模型", ["Text／Fixed／SASRec／GRU4Rec", "每种3个种子，每项6040用户", "同一真实目标、完整历史屏蔽", "生成模型Beam20；基线全物品打分"]),
             ("程序 · 分开评测与汇总", ["week7_evaluate.py / week7_baseline.py", "week7_report.py summarize", "逐用户验证Top-10与重算指标", "10000次用户配对bootstrap"]),
             ("得到 · 最终比较", ["06-evaluation/{generative,baselines}/模型/seed/", "summaries/：主表、分组、两组配对区间", "figures/test_comparison.png", "main_results.md：结论与证据边界"]),
         ], "Fixed NDCG@10=.1864，Text=.1594，GRU4Rec=.1847；相对Text约+17%，与GRU差值区间包含0。"),
         ("07  归档成一条可阅读的链路", [
             ("实验工作区 · 大型文件", ["work/week7/：向量与全部预测", "exported/week7/：冻结模型接口", "saved/week7/：量化/推荐权重", "运维与临时排队文件只留工作区"]),
             ("公开仓库 · 学习入口", ["README.md：问题、路线、最终效果", "docs/：入门、图解、协议、复现", "scripts/：按周命名的完整实现", "results/：分层原始证据与结论"]),
             ("终点 · 能说明什么", ["项目七周实验全部完成", "协同融合优于受控纯文本", "尚不能确认总体超越GRU4Rec", "长尾较弱，严格冷启动无有效目标"]),
         ], "终点检查：结果来自实际预测，test未用于筛选；公开结果与服务器归档一致，保留历史记录原始字节。"),
     ]),
]


def find_font(requested):
    candidates = ([requested] if requested else []) + [
        "C:/Windows/Fonts/msyh.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("Pass a Chinese font file with --font /path/to/font.ttc")


class Canvas:
    """Write identical text and geometry to a PNG and an editable SVG."""
    def __init__(self, width, height, font):
        self.image = Image.new("RGB", (width, height), COLORS["paper"])
        self.draw = ImageDraw.Draw(self.image)
        self.font_path = font
        self.fonts = {}
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                    '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif}</style>',
                    f'<rect width="100%" height="100%" fill="{COLORS["paper"]}"/>']

    def font(self, size):
        if size not in self.fonts:
            self.fonts[size] = ImageFont.truetype(self.font_path, size)
        return self.fonts[size]

    def rect(self, box, fill, radius=22, outline=None):
        self.draw.rounded_rectangle(box, radius, fill, outline=outline, width=2)
        x, y, right, bottom = box
        self.svg.append(f'<rect x="{x}" y="{y}" width="{right-x}" height="{bottom-y}" rx="{radius}" fill="{fill}" stroke="{outline or fill}" stroke-width="2"/>')

    def line(self, points, fill, width=5):
        self.draw.line(points, fill=fill, width=width)
        self.svg.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{fill}" stroke-width="{width}"/>')

    def text(self, position, value, size=36, fill=None):
        fill = fill or COLORS["ink"]
        self.draw.text(position, value, font=self.font(size), fill=fill, anchor="lt")
        x, y = position
        self.svg.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" dominant-baseline="text-before-edge">{escape(value)}</text>')

    def wrap(self, value, size, width):
        lines, current = [], ""
        for char in value:
            if self.draw.textlength(current + char, font=self.font(size)) > width:
                lines.append(current)
                current = char
            else:
                current += char
        return lines + ([current] if current else [])

    def save(self, stem):
        OUT.mkdir(parents=True, exist_ok=True)
        self.image.save(OUT / (stem + ".png"), optimize=True)
        (OUT / (stem + ".svg")).write_text("\n".join(self.svg + ["</svg>"]), encoding="utf-8")


def render_map(spec, font):
    stem, label, title, accent, subtitle, stages = spec
    measure = Canvas(W, 1, font)
    x0, gap, card_w = 260, 48, 1050
    rows = []
    for heading, cards, check in stages:
        wrapped = [[(line, measure.wrap(line, 35, card_w-72)) for line in lines]
                   for _, lines in cards]
        # One bullet is one file or mechanism; wrapped paths retain their meaning.
        heights = [112 + sum(len(parts)*52 + 15 for _, parts in items) for items in wrapped]
        check_lines = measure.wrap(check, 32, W-x0-160)
        rows.append((max(heights), wrapped, check_lines))
    height = 420 + sum(h + 210 + len(c)*45 for h, _, c in rows) + 210
    canvas = Canvas(W, height, font)
    canvas.rect((0, 0, W, 28), accent, radius=0)
    canvas.text((120, 88), label + "   /   FILE-LEVEL WORKFLOW", 36, accent)
    canvas.text((120, 151), title, 72)
    canvas.text((125, 270), subtitle, 35, COLORS["muted"])
    canvas.text((125, 337), "沿主干向下读；每行展开输入、程序与产物。路径相对实验根目录，processed/相对数据目录。", 30, COLORS["muted"])
    y = 440
    canvas.line([(132, y-35), (132, height-155)], COLORS["line"], 10)
    for index, ((heading, cards, check), (card_h, wrapped, check_lines)) in enumerate(zip(stages, rows), 1):
        canvas.rect((96, y-5, 168, y+67), accent, radius=36)
        canvas.text((113, y+9), str(index).zfill(2), 31, "#ffffff")
        canvas.line([(168, y+32), (235, y+32)], accent, 5)
        canvas.text((x0, y), heading, 46, accent)
        top = y+94
        for column, ((name, lines), bullets) in enumerate(zip(cards, wrapped)):
            x = x0 + column*(card_w+gap)
            fill = COLORS[("input", "code", "output")[column]]
            canvas.rect((x, top, x+card_w, top+card_h), "#ffffff", outline="#dce4ec")
            canvas.rect((x, top, x+card_w, top+74), fill, radius=18)
            canvas.text((x+32, top+18), name, 33)
            cy = top+106
            for _, parts in bullets:
                canvas.rect((x+32, cy+13, x+42, cy+23), accent, radius=4)
                for part in parts:
                    canvas.text((x+57, cy), part, 35)
                    cy += 52
                cy += 15
            if column < 2:
                mid = top+card_h//2
                end = x+card_w+gap-10
                canvas.line([(x+card_w+10, mid), (end, mid)], accent, 4)
                canvas.line([(end-12, mid-10), (end, mid), (end-12, mid+10)], accent, 4)
        cy = top+card_h+28
        for line in check_lines:
            canvas.text((x0+8, cy), line, 32, COLORS["muted"])
            cy += 45
        y = cy+88
    canvas.text((120, height-105), "生成式推荐学习仓库  ·  图中数字来自已完成实验；原始日志与冻结记录保留运行时命名。", 30, COLORS["muted"])
    canvas.save(stem)
    print(f"{stem}: {W}×{height} PNG + SVG")


def overview(font):
    canvas = Canvas(1800, 720, font)
    canvas.text((65, 40), "电影先有编号，模型再学会生成编号", 54)
    canvas.text((68, 122), "先给电影编码，再让模型学习观看顺序；推理时把编号查回真实电影。", 28, COLORS["muted"])
    cards = [
        (65, "① 认识电影", ["标题、类型＋训练集观看行为", "变成一组描述电影的数字"], "#fff1da"),
        (650, "② 编成语义ID", ["相似电影可以共享编号前缀", "例如 (37, 128, 5, 0)"], "#e7efff"),
        (1235, "③ 逐位生成", ["历史电影ID → 生成下一部ID", "合法编号 → 真实电影Top-10"], "#e0f3e9"),
    ]
    for x, title, body, color in cards:
        canvas.rect((x, 215, x+500, 470), color)
        canvas.text((x+27, 245), title, 38)
        for i, line in enumerate(body):
            canvas.text((x+27, 327+i*63), line, 27)
    for x in (585, 1170):
        canvas.line([(x, 340), (x+42, 340)], "#5b7182", 6)
        canvas.line([(x+28, 327), (x+42, 340), (x+28, 353)], "#5b7182", 6)
    canvas.rect((65, 530, 1735, 650), "#ffffff", outline="#dce4ec")
    canvas.text((95, 552), "训练：用真实下一部电影教模型逐位预测。推荐：模型接着自己的前缀生成，查回电影。", 30)
    canvas.text((95, 603), "编号仅为示意  ·  非法编号和完整历史中看过的电影会被排除  ·  最终效果独立测试", 25, COLORS["muted"])
    canvas.save("overview")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", help="Chinese .ttf/.ttc/.otf font file")
    args = parser.parse_args()
    font = find_font(args.font)
    for spec in MAPS:
        render_map(spec, font)
    overview(font)


if __name__ == "__main__":
    main()
