"""Create reader-facing documents and a SHA-256 checked whitelist release ZIP."""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path
import re
import textwrap
import zipfile

from week7_common import ROOT, VARIANTS, read_json, sha256, write_json

SCRIPTS = (
    "week7_common.py", "week7_prepare.py", "week7_train.py",
    "week7_evaluate.py", "week7_baseline.py", "week7_report.py",
    "week7_release.py", "run_week7.py",
)


def save(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def refresh_readme(path, block):
    path = Path(path)
    start, end = "<!-- WEEK7:START -->", "<!-- WEEK7:END -->"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    section = start + "\n" + block.strip() + "\n" + end
    if start in current:
        if end not in current:
            raise ValueError(f"missing end marker: {path}")
        current = re.sub(re.escape(start) + r".*?" + re.escape(end),
                         lambda _: section, current, flags=re.S)
    else:
        current = current.rstrip() + "\n\n" + section + "\n"
    path.write_text(current, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--license", choices=("MIT",), required=True)
    parser.add_argument("--scaffold", action="store_true")
    args = parser.parse_args()
    if not args.url.startswith("https://github.com/"):
        raise ValueError("use the public HTTPS GitHub repository URL")
    if not args.author.strip() or "\n" in args.author:
        raise ValueError("supply the repository copyright holder")
    save("docs/learning-guide.md", """
        # 生成式推荐学习路径

        本项目用 MovieLens 1M 演示“电影如何变成可生成的标识符”。

        1. 电影标题和类型由冻结的 MiniLM 编码成384维文本向量。
        2. Week 7 额外仅从 train 构造用户—电影稀疏矩阵，SVD提取64维协同向量。
        3. 拼接两类归一化向量后再归一化，得到448维输入。
        4. RQ-VAE先预训练连续AE，再在32维潜空间中逐层K-Means初始化码本，
           最后联合优化重建、码本和commitment损失。
        5. 三个256大小码本产生三位编号，第四位用于碰撞消解。
        6. 四个位置使用互不重叠的Token区间：token=3+256*position+code。
           四位ID(37,128,5,0)对应[40,387,520,771]；这只是编码示例。
        7. 用户最近20部电影形成最多81个Encoder Token；目标是4个Token加EOS。
           Decoder输入为BOS和右移标签，训练使用逐位置交叉熵。
        8. 推理从BOS开始逐位生成；动态Trie排除不存在的ID和完整历史中的已见电影。
        9. 生成结果逆映射为电影，评测Recall/NDCG@5/10。

        文本/协同向量、RQ码字和T5词嵌入是三种不同表示。
        RQ-VAE与T5分阶段训练；推荐损失没有回传到RQ-VAE。
        旧系统已在T5训练时学习用户行为，Week 7把协同信息提前用于ID构造。
        新ID改变Token含义，因此对应的T5需要重新训练。
    """)
    save("docs/experiment-protocol.md", """
        # 实验协议

        数据沿用既有5-core与leave-one-out冻结划分。训练6040条用户序列，
        生成模型使用981491个在线全前缀样本；valid/test各6040名用户。

        协同特征只读train；物品训练频次决定融合权重和分组。
        Text、Fixed、Fixed-Match、Adaptive输入均为448维，
        RQ-VAE为3层256码本、32维潜空间、量化seed42。
        Fixed-Match匹配Adaptive的平均协同平方范数占比。

        T5结构和1027词表沿用原配置；历史20、有效batch256、
        peak lr3e-4、warmup1500、cosine floor0.05、最多60k步。
        每5k步按约束valid NDCG@10选模；30k后启用patience4。
        训练方案对比同时改变预算和日程，不能只归因为训练步数。

        四组seed42只用于valid筛选。Text与胜出融合方案补齐seed43/44。
        完成六个正式运行并冻结后才开启test；test不参与超参选择。
        主评测为全目录、完整历史屏蔽、beam20、Top-10。

        三种子标准差使用ddof=0。用户配对bootstrap先平均每个用户的三个seed差值，
        再以用户重采样10000次；其区间不包含训练随机性的全部不确定性。
        strict-cold为N/A(n=0)，few-shot<=20仅是稀疏物品代理。
        h20受控基线统一验证选模；旧h200结果作为历史实用配置参考。
    """)
    save("docs/reproduction.md", """
        # 复现入口

        实验目录采用相对路径。完整实验需要冻结的exported/六文件和item_emb.npy；
        原始数据预处理、文本编码和量化流程见仓库README及Week 1–6结果索引。
        CPU测试不要求真实数据，也不下载预训练模型。

        ## CPU快速检查

            python -m unittest discover -s tests -p "test_week7.py" -v
            python -m compileall -q scripts

        ## CPU准备

            python scripts/run_week7.py init
            python scripts/run_week7.py features
            python scripts/run_week7.py tokenize --variant text
            python scripts/run_week7.py tokenize --variant fixed
            python scripts/run_week7.py tokenize --variant fixed_match
            python scripts/run_week7.py tokenize --variant adaptive
            python scripts/run_week7.py smoke --stop_after 2
            python scripts/run_week7.py smoke --resume --stop_after 4

        ## 单卡训练和恢复

            python scripts/run_week7.py train --variant text --seed 42 --gpu 0
            python scripts/run_week7.py train --variant text --seed 42 --gpu 0 --resume

        物理卡号由执行者检查后填写。启动器发现计算进程或显存占用会退出；
        它不会排队或结束他人进程。CPU准备、测试和文档不需要空卡。

        ## 完整实验顺序

        四组seed42训练完成后运行select；Text与selected补齐43/44。
        两种h20基线各训练42/43/44；随后运行全部正式test并summarize。
        每个训练、评测命令都支持--dry_run显示底层命令。
        一张24GB卡运行一个训练进程；OOM时把batch64/accum4改为32/8。
        修改batch配置需对比较组保持一致，恢复已有运行时必须使用原配置。

        真实GPU耗时和磁盘量以本次日志为准，不把未知耗时写成保证。
        大checkpoint保留于实验服务器，公开仓库保留配置、哈希和轻量结果。
    """)
    save("requirements-week7.txt", """
        # Install a suitable CPU/CUDA torch wheel first. Existing server uses its current environment.
        numpy>=1.26,<2
        scipy>=1.10,<1.18
        scikit-learn>=1.4,<2
        PyYAML>=6,<7
        matplotlib>=3.8,<4
        transformers>=5.19,<6
        safetensors
    """)
    save(".github/workflows/ci.yml", """
        name: week7-cpu-tests
        on: [push, pull_request]
        jobs:
          cpu-tests:
            runs-on: ubuntu-latest
            steps:
              - uses: actions/checkout@v4
              - uses: actions/setup-python@v5
                with:
                  python-version: '3.11'
              - run: python -m pip install "numpy<2" scipy scikit-learn PyYAML
              - run: python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
              - run: python -m pip install "transformers>=5.19,<6"
              - run: python -m compileall -q scripts
              - run: python -m unittest discover -s tests -p "test_week7.py" -v
    """)
    save("CITATION.cff", f"""
        cff-version: 1.2.0
        message: "If you use this learning repository, please cite this project and its referenced papers."
        title: "Generative Recommendation: A Reproducible Learning Repository"
        authors:
          - name: "{args.author.replace(chr(34), chr(39))}"
        repository-code: "{args.url}"
        version: 0.1.0
        license: MIT
    """)
    license_text = f"""
        MIT License

        Copyright (c) {datetime.date.today().year} {args.author}

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
    """
    if Path("LICENSE").exists():
        if Path("LICENSE").read_text(encoding="utf-8").strip() != textwrap.dedent(license_text).strip():
            raise ValueError("existing LICENSE differs; preserve its existing terms")
    else:
        save("LICENSE", license_text)
    ignore = Path(".gitignore")
    current = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
    entries = ("/work/", "/exported/", "/dist/", "*.pt", "*.npz", "*.npy",
               "/results/week7-improvement/05-formal-training/smoke_*")
    for entry in entries:
        if entry not in current.splitlines():
            current += "\n" + entry
    ignore.write_text(current.rstrip() + "\n", encoding="utf-8")
    if args.scaffold:
        print("CPU documentation, CI, dependencies and chosen license generated")
        return
    selection = read_json(ROOT / "selection.json")
    read_json(ROOT / "06-evaluation/summary.json")
    read_json(ROOT / "06-evaluation/baseline_h20_summary.json")
    if not (ROOT / "01-diagnostics/search_diagnosis.md").exists():
        raise FileNotFoundError("finish the fixed-user Beam/exact diagnosis")
    ci = read_json(ROOT / "06-evaluation/paired_bootstrap.json")
    winner = selection["selected"]
    delta = ci["all"]["ndcg@10"]
    lo, hi = delta["ci95"]
    statement = (
        "在当前三套已训练模型上观察到稳定正向用户配对差异。"
        if lo > 0 else
        "当前结果未证明稳定正向差异，需结合单种子波动与分组结果解释。"
    )
    for page in ("README.md", "results/README.md"):
        if not Path(page).exists():
            raise FileNotFoundError(f"copy the current public {page} to the server first")
    main_readme = Path("README.md").read_text(encoding="utf-8")
    main_readme = re.sub(
        r"(?m)^\| 第 7 周 \|.*$",
        "| 第 7 周 | 文本—协同ID消融、三种子验证与仓库整理 | 已记录实测结果 |",
        main_readme,
    )
    main_readme = re.sub(
        r"(?m)^\| Week 7 \|.*$",
        "| Week 7 | 文本向量与训练交互 | 四组ID消融、三种子与搜索诊断 | 可复现实验表与学习文档 |",
        main_readme,
    )
    Path("README.md").write_text(main_readme, encoding="utf-8")
    result_readme = Path("results/README.md").read_text(encoding="utf-8")
    result_readme = re.sub(
        r"(?m)^\| Week 7 \|.*$",
        "| Week 7 | [改进与消融](week7-improvement/README.md) | 已记录实测结果 |",
        result_readme,
    )
    Path("results/README.md").write_text(result_readme, encoding="utf-8")
    save(ROOT / "06-evaluation/limitations.md", f"""
        # 实验边界

        最终候选：{winner}。{statement}

        - 仅在MovieLens 1M验证，不能代表其他数据或生产环境。
        - 协同融合已有相关研究，本项目属于可复现实验与启发式探索。
        - SVD不保留交互顺序；训练频次不是协同向量可靠性的严格估计。
        - RQ-VAE重建与下一物品推荐仍是两阶段目标，未做端到端联合优化。
        - strict-cold目标数为0；少样本test目标仅72个，分组结论存在较大不确定性。
        - seed42的四组结构消融是探索性筛选，只有Text与最终方案具备三种子正式结果。
        - 用户bootstrap置信区间以当前已训练模型为条件，不涵盖全部训练随机性。
        - 旧30k与新60k训练方案同时改变预算与学习率轨迹。
        - 精确目录打分只在固定512名valid用户诊断，不能当作线上推理吞吐或NDCG上界。
    """)
    index = [
        "# Week 7：协同信号与语义标识符", "",
        "| 目录 | 内容 |", "|---|---|",
        "| 01-diagnostics | 训练曲线、固定用户Beam/exact诊断 |",
        "| 02-embeddings | train-only协同表示与权重元数据 |",
        "| 03-semantic-ids | 四组量化日志、质量统计与接口哈希 |",
        "| 04-screening | seed42验证集消融与方案选择 |",
        "| 05-formal-training | 三种子历史与最佳验证指标 |",
        "| 06-evaluation | 正式test、分组、样例、bootstrap与主表 |",
        "| 07-release | 同步白名单与SHA-256清单 |",
        "", f"最终候选为{winner}。{statement}",
        f"NDCG@10相对Text差值为{delta['delta']:+.6f}，95% CI [{lo:+.6f}, {hi:+.6f}]。",
        "", "完整数值见[主表](06-evaluation/main_results.md)。"
    ]
    save(ROOT / "README.md", "\n".join(index))
    section = (
        "## Week 7：协同信号与语义标识符\n\n"
        f"最终候选：{winner}。{statement}\n\n"
        f"test NDCG@10差值：{delta['delta']:+.6f}，"
        f"用户配对95% CI [{lo:+.6f}, {hi:+.6f}]。\n\n"
        "[结果与消融](results/week7-improvement/README.md) · "
        "[学习路径](docs/learning-guide.md) · "
        "[复现说明](docs/reproduction.md) · "
        "[协议](docs/experiment-protocol.md)"
    )
    refresh_readme("README.md", section)
    refresh_readme("results/README.md",
                   "## Week 7\n\n[改进、消融与结项产物](week7-improvement/README.md)")
    files = [Path("scripts") / name for name in SCRIPTS]
    files += [Path("tests/test_week7.py")]
    files += list(Path("config/week7").glob("*.yaml"))
    files += list(Path("docs").glob("*.md"))
    files += [Path(name) for name in (
        "README.md", "results/README.md", ".gitignore",
        "requirements-week7.txt", "CITATION.cff", "LICENSE",
        ".github/workflows/ci.yml",
    )]
    files += [p for p in ROOT.rglob("*")
              if p.is_file() and "07-release" not in p.parts
              and not p.name.startswith("smoke_")]
    for path in files:
        if not path.exists() or path.stat().st_size > 12 * 1024 * 1024:
            raise ValueError(f"missing or unexpectedly large public file: {path}")
    # Preserve raw experimental evidence; review logs for credentials before publication.
    write_json(ROOT / "07-release/manifest.json", {
        p.as_posix(): sha256(p) for p in sorted(set(files))
    })
    files.append(ROOT / "07-release/manifest.json")
    Path("dist").mkdir(exist_ok=True)
    with zipfile.ZipFile("dist/week7-public.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(set(files)):
            bundle.write(path, path.as_posix())
    print("dist/week7-public.zip")


if __name__ == "__main__":
    main()