"""Curve extraction, valid-only selection, paired inference, result tables."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from week7_common import (
    METRICS, PROTOCOL, ROOT, VARIANTS, artifact_hash, object_hash, ranking_metrics,
    read_json, read_jsonl, require_test_unlock, sha256, write_json,
)


def curves():
    paths = list(Path("results/week5-6-generative/03-formal-training").glob("*history.jsonl"))
    paths += list((ROOT / "05-formal-training").glob("*history.jsonl"))
    records = []
    for path in paths:
        for row in read_jsonl(path):
            if row.get("type") == "valid":
                records.append({
                    "run": path.stem, "step": row["step"],
                    "valid_ndcg10": row["metrics"]["all"]["ndcg@10"],
                })
    out = ROOT / "01-diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "training_curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("run", "step", "valid_ndcg10"))
        writer.writeheader()
        writer.writerows(records)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in sorted({r["run"] for r in records}):
        values = sorted((r for r in records if r["run"] == name),
                        key=lambda r: r["step"])
        ax.plot([r["step"] for r in values],
                [r["valid_ndcg10"] for r in values], label=name)
    ax.set(xlabel="optimizer step", ylabel="valid NDCG@10")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=160)
    plt.close(fig)
    print(f"{len(records)} validation points exported")


def select():
    path = ROOT / "selection.json"
    if path.exists():
        raise ValueError("selection already frozen; do not overwrite after test")
    records = {}
    for name in VARIANTS:
        meta = read_json(ROOT / "05-formal-training" / f"{name}_s42_best.json")
        if meta["smoke"] or not meta["finished"]:
            raise ValueError(f"unfinished screening run: {name}")
        if meta["artifact_hash"] != artifact_hash(f"exported/week7/{name}"):
            raise ValueError(f"changed artifacts: {name}")
        records[name] = meta
    # Representation paths and variant differ; all training/model controls must agree.
    controls = [{"model": r["config"]["model"], "data": r["config"]["data"],
                 "train": r["config"]["train"], "decode": r["config"]["decode"]}
                for r in records.values()]
    if len({object_hash(x) for x in controls}) != 1:
        raise ValueError("screening controls differ")
    scores = {name: r["metrics"]["all"]["ndcg@10"] for name, r in records.items()}
    # Stable preference on exact ties: fixed, then fixed_match, then adaptive.
    winner = max(("fixed", "fixed_match", "adaptive"), key=lambda x: scores[x])
    protocol = {
        "protocol_version": PROTOCOL, "selected": winner,
        "selection_split": "valid", "metric": "ndcg@10", "scores": scores,
        "delta_vs_text": scores[winner] - scores["text"],
        "passes_practical_screen": scores[winner] - scores["text"] >= 0.003,
        "seeds": [42, 43, 44], "history": 20, "max_steps": 60000,
        "beam": 20, "mask_full_history": True,
        "controls": controls[0],
        "artifact_hashes": {name: r["artifact_hash"] for name, r in records.items()},
    }
    write_json(path, protocol)
    write_json(ROOT / "04-screening/ablation.json", {
        "scope": "valid, seed42; candidate selection, not a robust test claim",
        "metrics": {name: r["metrics"] for name, r in records.items()},
    })
    print(protocol)


def paired_bootstrap(differences, resamples=10000, seed=42):
    differences = np.asarray(differences, dtype=np.float64)
    if not len(differences):
        return {"n_users": 0, "delta": None, "ci95": None}
    rng = np.random.default_rng(seed)
    means = []
    # Batches cap memory at about 64 * n_users floats.
    for start in range(0, resamples, 64):
        count = min(64, resamples - start)
        sample = rng.integers(0, len(differences), size=(count, len(differences)))
        means.extend(differences[sample].mean(axis=1).tolist())
    return {
        "n_users": len(differences), "delta": float(differences.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "bootstrap_unit": "user; fixed fitted models; seed-average paired difference",
    }


def diagnostics():
    from week7_common import ranking_metrics
    folder = ROOT / "01-diagnostics"
    paths = [folder / f"historical_30k_s42_valid_beam{b}_n512.json"
             for b in (10, 20, 50, 100)]
    paths.append(folder / "historical_30k_s42_valid_exact_n512.json")
    records = [read_json(path) for path in paths]
    if len({object_hash(r["user_ids"]) for r in records}) != 1:
        raise ValueError("diagnostic user lists differ")
    if len({object_hash(r["checkpoint_sha256"]) for r in records}) != 1:
        raise ValueError("diagnostic checkpoints differ")
    exact_path = Path("work/week7/predictions/historical_30k_s42_valid_exact_n512.jsonl")
    exact = {r["user"]: r for r in read_jsonl(exact_path)}
    lines = ["# 固定512名验证用户的搜索诊断", "",
             "| 方法 | NDCG@10 | Recall@10 | 总秒数 | MiB | 与exact的Top-10重合率 |",
             "|---|---:|---:|---:|---:|---:|"]
    for path, record in zip(paths, records):
        pred_path = Path("work/week7/predictions") / (path.stem + ".jsonl")
        pred = read_jsonl(pred_path)
        overlap = np.mean([len(set(r["ranked_items"]) &
                               set(exact[r["user"]]["ranked_items"])) / 10 for r in pred])
        label = "exact" if record["beam"] is None else f"beam{record['beam']}"
        metrics = record["metrics"]["all"]
        lines.append(
            f"| {label} | {metrics['ndcg@10']:.6f} | {metrics['recall@10']:.6f} | "
            f"{record['seconds']:.2f} | {record['peak_gpu_mib']:.1f} | {overlap:.4f} |"
        )
    lines += ["", "所有方法使用相同用户、checkpoint和完整历史屏蔽。",
              "exact按完整ID的原始Token对数概率求和排序，包含EOS，不做合法分支重归一化。",
              "exact是模型概率排序参考；NDCG可能高于或低于Beam结果。",
              "耗时为预热后的批处理整段时间，不能当作在线服务P95延迟。"]
    (folder / "search_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(folder / "search_diagnosis.md")


def summarize():
    selection = require_test_unlock("text", 42)
    variants = ("text", selection["selected"])
    records, predictions = {}, {}
    for name in variants:
        for seed in (42, 43, 44):
            tag = f"{name}_s{seed}_test_beam20_n6040"
            result = read_json(ROOT / "06-evaluation" / f"{tag}.json")
            if result["artifact_hash"] != selection["artifact_hashes"][name]:
                raise ValueError("formal result artifacts mismatch")
            if result["n_users"] != 6040 or any(result["diagnostics"].values()):
                raise ValueError("formal evaluation failed count or legality gate")
            pred = read_jsonl(Path("work/week7/predictions") / f"{tag}.jsonl")
            if len(pred) != 6040 or len({r["user"] for r in pred}) != 6040:
                raise ValueError("missing or duplicate prediction rows")
            records[name, seed] = result
            predictions[name, seed] = {int(r["user"]): r for r in pred}
    summaries = {}
    for name in variants:
        summaries[name] = {}
        for group in records[name, 42]["metrics"]:
            summaries[name][group] = {"n": records[name, 42]["metrics"][group]["n"]}
            for metric in METRICS:
                values = [records[name, seed]["metrics"][group][metric]
                          for seed in (42, 43, 44)]
                summaries[name][group][metric] = (
                    None if values[0] is None else
                    {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}
                )
    from generative_common import SemanticArtifacts, build_groups
    a = SemanticArtifacts("exported")
    groups, _ = build_groups(read_jsonl("exported/train.semantic.jsonl"), a)
    groups = {"all": set(a.item_to_code), **groups}
    ci = {}
    for group, items in groups.items():
        ci[group] = {}
        for metric in METRICS:
            differences = []
            for user in sorted(predictions["text", 42]):
                target = predictions["text", 42][user]["target_item"]
                if target not in items:
                    continue
                ds = []
                for seed in (42, 43, 44):
                    first = predictions["text", seed][user]
                    second = predictions[selection["selected"], seed][user]
                    if first["target_item"] != second["target_item"]:
                        raise ValueError("paired target mismatch")
                    ds.append(
                        ranking_metrics(second["ranked_items"], target)[metric]
                        - ranking_metrics(first["ranked_items"], target)[metric]
                    )
                differences.append(np.mean(ds))
            ci[group][metric] = paired_bootstrap(differences)
    write_json(ROOT / "06-evaluation/summary.json", summaries)
    write_json(ROOT / "06-evaluation/paired_bootstrap.json", ci)
    lines = [
        "# 第七周：协同信号与语义标识符", "",
        "MovieLens 1M；三个训练种子；全量排序；屏蔽完整历史；总体标准差 ddof=0。",
        "", "| 模型 | 分组 | n | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, by_group in summaries.items():
        for group, values in by_group.items():
            cells = []
            for metric in METRICS:
                value = values[metric]
                cells.append("N/A" if value is None else
                             f"{value['mean']:.4f} ± {value['std']:.4f}")
            lines.append(f"| {name} | {group} | {values['n']} | "
                         + " | ".join(cells) + " |")
    lines += ["", "## Seed42 验证集结构消融", "",
              "| 表示 | valid NDCG@10 |", "|---|---:|"]
    for name, score in selection["scores"].items():
        lines.append(f"| {name} | {score:.6f} |")
    delta = ci["all"]["ndcg@10"]
    lo, hi = delta["ci95"]
    lines += [
        "", f"最终候选：{selection['selected']}；test NDCG@10 差值 "
        f"{delta['delta']:+.6f}，用户配对 bootstrap 95% CI [{lo:+.6f}, {hi:+.6f}]。",
        "", "该区间只描述当前三个已训练模型条件下的用户抽样不确定性，"
        "不能代替更多训练种子或跨数据集验证。",
        "", "## 结果边界", "",
        "strict-cold 为 N/A（n=0）。少样本指标不等同于真正新物品冷启动。",
        "旧 30k 与新 60k 同时改变训练预算和学习率轨迹，属于训练方案对比。",
        "精确目录打分是模型概率排序参考，不是 NDCG 上界。",
        "频次自适应属于启发式；未取得三种子证据的结构消融仅作探索性结论。",
    ]
    out = ROOT / "06-evaluation/main_results.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Baseline results are summarized separately and never replace the old h200 table.
    baseline = []
    for model in ("sasrec", "gru4rec"):
        paths = [ROOT / "06-evaluation" / f"{model}_h20_s{s}_test.json"
                 for s in (42, 43, 44)]
        if not all(p.exists() for p in paths):
            raise FileNotFoundError(f"finish all h20 baseline test runs: {model}")
        runs = [read_json(p) for p in paths]
        baseline.append({
            "model": model, "history": 20,
            "metrics": {m: {"mean": float(np.mean([r["metrics"]["all"][m] for r in runs])),
                            "std": float(np.std([r["metrics"]["all"][m] for r in runs]))}
                        for m in METRICS},
        })
    write_json(ROOT / "06-evaluation/baseline_h20_summary.json", baseline)
    with out.open("a", encoding="utf-8") as handle:
        handle.write("\n## 历史长度20的受控对比\n\n")
        handle.write("| 模型 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        combined = baseline + [
            {"model": name, "metrics": summaries[name]["all"]} for name in variants
        ]
        for row in combined:
            cells = [f"{row['metrics'][m]['mean']:.4f} ± {row['metrics'][m]['std']:.4f}"
                     for m in METRICS]
            handle.write("| " + row["model"] + " | " + " | ".join(cells) + " |\n")
        handle.write("\n## 旧配置效果参考\n\n")
        handle.write("旧h200基线NDCG@10：SASRec 0.1637±0.0034，GRU4Rec 0.1882±0.0009。"
                     "旧h20生成模型为0.1180±0.0092。"
                     "旧基线checkpoint选模与本周的对齐口径存在差异，"
                     "该表只作已有配置实用效果参考，不能冒充受控架构比较。\n")
    print(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("curves", "diagnostics", "select", "summarize"))
    args = parser.parse_args()
    {"curves": curves, "diagnostics": diagnostics,
     "select": select, "summarize": summarize}[args.action]()


if __name__ == "__main__":
    main()