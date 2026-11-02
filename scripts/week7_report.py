"""Curve extraction, valid-only selection, paired inference, result tables."""
from __future__ import annotations
from week1_paths import result_path, output_path

import argparse
import csv
from pathlib import Path

import numpy as np

from week7_common import (
    METRICS, PROTOCOL, ROOT, VARIANTS, artifact_hash, object_hash, ranking_metrics,
    read_json, read_jsonl, require_test_unlock, sha256, write_json,
)


def curves():
    # Both the public archive and the experiment workspace use the same layout.
    # Deduplicate historical names if a legacy root is also present.
    historical = {}
    for folder in (Path("results/week5-6-generative/03-formal-training"),
                   Path("results/week5-6-generative")):
        for path in sorted(folder.rglob("t5_sid_seed*_history.jsonl")):
            historical.setdefault(path.name, path)
    paths = list(historical.values())
    for path in sorted((ROOT / "05-formal-training").rglob("*_history.jsonl")):
        meta_path = path.with_name(path.name.replace("_history.jsonl", "_best.json"))
        if not meta_path.exists():
            continue
        meta = read_json(meta_path)
        if meta.get("finished") and not meta.get("smoke"):
            paths.append(path)
    records = []
    for path in paths:
        for row in read_jsonl(path):
            if row.get("type") == "valid":
                records.append({
                    "run": path.stem.removesuffix("_history"), "step": row["step"],
                    "valid_ndcg10": row["metrics"]["all"]["ndcg@10"],
                })
    out = ROOT / "01-diagnostics"
    records.sort(key=lambda row: (row["run"], row["step"]))
    out.mkdir(parents=True, exist_ok=True)
    with output_path(out / "training_curves.csv").open("w", encoding="utf-8", newline="") as handle:
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
    fig.savefig(output_path(out / "training_curves.png"), dpi=160)
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
    output_path(folder / "search_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(folder / "search_diagnosis.md")


GROUP_LABELS = {"all": "总体", "few_shot_le20": "少样本（频次≤20）",
                "tail_bottom80": "长尾（后80%物品）", "head_top20": "头部（前20%物品）",
                "strict_cold": "严格冷启动"}
MODEL_LABELS = {"sasrec": "SASRec", "gru4rec": "GRU4Rec", "text": "T5 · Text",
                "fixed": "T5 · Fixed", "fixed_match": "T5 · Fixed-Match",
                "adaptive": "T5 · Adaptive"}


def _test_tag(name, seed):
    if name in ("sasrec", "gru4rec"):
        return f"{name}_h20_s{seed}_test"
    return f"{name}_s{seed}_test_beam20_n6040"


def _test_inputs(models):
    paths = {}
    for name in models:
        for seed in (42, 43, 44):
            tag = _test_tag(name, seed)
            files = (ROOT / "06-evaluation" / f"{tag}.json",
                     ROOT / "06-evaluation" / f"{tag}_samples.json",
                     Path("work/week7/predictions") / f"{tag}.jsonl")
            files = tuple(result_path(path) for path in files)
            for path in files:
                if not path.is_file():
                    raise FileNotFoundError(f"finish all twelve frozen test evaluations: {path}")
            paths[name, seed] = files
    return paths


def _group_summary(records, name):
    result = {}
    for group in GROUP_LABELS:
        result[group] = {"n": records[name, 42]["metrics"][group]["n"]}
        for metric in METRICS:
            values = [records[name, seed]["metrics"][group][metric] for seed in (42, 43, 44)]
            result[group][metric] = (None if values[0] is None else
                {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))})
    return result


def _format_score(value):
    return "N/A" if value is None else f"{value['mean']:.4f} ± {value['std']:.4f}"


def _plot_test_results(records, summaries, models, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ("#8797a6", "#546e94", "#dc9b43", "#238b7d")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    positions = np.arange(len(models))
    for axis, group, title in zip(axes, ("all", "tail_bottom80"), ("Overall", "Tail items")):
        means = [summaries[name][group]["ndcg@10"]["mean"] for name in models]
        stds = [summaries[name][group]["ndcg@10"]["std"] for name in models]
        axis.bar(positions, means, color=colors, width=0.62, alpha=0.85)
        axis.errorbar(positions, means, yerr=stds, fmt="none", color="#243447", capsize=4)
        for offset, seed in zip((-0.08, 0, 0.08), (42, 43, 44)):
            values = [records[name, seed]["metrics"][group]["ndcg@10"] for name in models]
            axis.scatter(positions + offset, values, color="#243447", s=15, zorder=3)
        for x, mean, std in zip(positions, means, stds):
            axis.text(x, mean + std + 0.004, f"{mean:.4f}", ha="center", fontsize=9)
        axis.set_xticks(positions, [MODEL_LABELS[name].replace("T5 · ", "T5 ") for name in models])
        axis.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.24)
        axis.set_ylabel("NDCG@10")
        axis.set_title(f"{title} (n={summaries[models[0]][group]['n']:,})")
        axis.yaxis.grid(True, alpha=0.18)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Full-catalog test | history 20 | seeds 42, 43, 44", fontsize=12)
    fig.text(0.5, 0.025, "Bars: mean; error bars: population standard deviation; dots: individual seeds.",
             ha="center", fontsize=9, color="#44515d")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def summarize():
    selection = require_test_unlock("text", 42)
    variants = ("text", selection["selected"])
    models = ("sasrec", "gru4rec", *variants)
    # Check every input before publishing any summary of a partial experiment.
    paths = _test_inputs(models)
    from week5_6_common import SemanticArtifacts, aggregate_rows, build_groups
    artifacts = SemanticArtifacts("exported")
    groups, _ = build_groups(read_jsonl("exported/train.semantic.jsonl"), artifacts)
    expected = {int(row["user"]): {
        "target": artifacts.code_item(row["target"]),
        "seen": {artifacts.code_item(code) for code in row["history"]},
    } for row in read_jsonl("exported/test.semantic.jsonl")}
    catalog = set(artifacts.item_to_code)
    records, predictions = {}, {}
    for (name, seed), (result_path, sample_path, prediction_path) in paths.items():
        record, pred = read_json(result_path), read_jsonl(prediction_path)
        if (record["split"] != "test" or record["seed"] != seed or record["n_users"] != 6040
                or not record["full_sort"] or not record["mask_full_history"]):
            raise ValueError(f"test protocol mismatch: {name} seed{seed}")
        if len(pred) != 6040 or {int(row["user"]) for row in pred} != set(expected):
            raise ValueError("missing or duplicate prediction users")
        for row in pred:
            truth, ranked = expected[int(row["user"])], row["ranked_items"]
            if (row["target_item"] != truth["target"] or len(ranked) != 10 or len(set(ranked)) != 10
                    or set(ranked) - catalog or set(ranked) & truth["seen"]):
                raise ValueError(f"target or Top-10 mismatch: {name} seed{seed}")
        if record["metrics"] != aggregate_rows(pred, groups) or read_json(sample_path) != pred[:20]:
            raise ValueError(f"saved metrics or samples differ from full predictions: {name} seed{seed}")
        if name in variants:
            if (record["variant"] != name or record["beam"] != 20
                    or record["artifact_hash"] != selection["artifact_hashes"][name]
                    or any(record["diagnostics"].values())):
                raise ValueError("generative frozen protocol mismatch")
        elif record["model"].lower() != name or record["max_history"] != 20:
            raise ValueError("baseline identity/history mismatch")
        records[name, seed] = record
        predictions[name, seed] = {int(row["user"]): row for row in pred}
    all_summaries = {name: _group_summary(records, name) for name in models}
    ci = {}
    for group, items in {"all": catalog, **groups}.items():
        ci[group] = {}
        for metric in METRICS:
            differences = []
            for user, row in sorted(predictions["text", 42].items()):
                target = row["target_item"]
                if target in items:
                    differences.append(np.mean([
                        ranking_metrics(predictions[selection["selected"], seed][user]["ranked_items"], target)[metric]
                        - ranking_metrics(predictions["text", seed][user]["ranked_items"], target)[metric]
                        for seed in (42, 43, 44)
                    ]))
            ci[group][metric] = paired_bootstrap(differences)
    baseline_differences = [np.mean([
        ranking_metrics(predictions[selection["selected"], seed][user]["ranked_items"], row["target_item"])["ndcg@10"]
        - ranking_metrics(predictions["gru4rec", seed][user]["ranked_items"], row["target_item"])["ndcg@10"]
        for seed in (42, 43, 44)
    ]) for user, row in sorted(predictions["text", 42].items())]
    baseline_ci = {"candidate": selection["selected"], "reference": "gru4rec",
                   "metric": "ndcg@10", "group": "all", "seeds": [42, 43, 44],
                   "resamples": 10000, **paired_bootstrap(baseline_differences)}
    folder = ROOT / "06-evaluation"
    write_json(folder / "summary.json", {name: all_summaries[name] for name in variants})
    write_json(folder / "paired_bootstrap.json", ci)
    write_json(folder / "gru4rec_comparison.json", baseline_ci)
    write_json(folder / "baseline_h20_summary.json", [
        {"model": name, "history": 20, "metrics": all_summaries[name]["all"],
         "groups": all_summaries[name]} for name in ("sasrec", "gru4rec")
    ])
    with output_path(folder / "test_results.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["model", "group", "n"] + [f"{metric}_{part}" for metric in METRICS for part in ("mean", "std")]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in models:
            for group, values in all_summaries[name].items():
                row = {"model": name, "group": group, "n": values["n"]}
                row.update({f"{metric}_{part}": None if values[metric] is None else values[metric][part]
                            for metric in METRICS for part in ("mean", "std")})
                writer.writerow(row)
    _plot_test_results(records, all_summaries, models, output_path(folder / "test_comparison.png"))
    delta = ci["all"]["ndcg@10"]
    relative = delta["delta"] / all_summaries["text"]["all"]["ndcg@10"]["mean"] * 100
    lo, hi = delta["ci95"]
    selected = MODEL_LABELS[selection["selected"]]
    lines = ["# 🎯 协同信号与语义标识符：最终结果", "",
             "MovieLens 1M；6,040名用户、3,416部电影；历史长度20；全目录排序并屏蔽完整历史。"
             "训练种子42/43/44，报告均值与总体标准差（ddof=0）。", "",
             f"**{selected}相对Text的test NDCG@10提高约{relative:.1f}%**；差值{delta['delta']:+.6f}，"
             f"用户配对bootstrap 95%区间为[{lo:+.6f}, {hi:+.6f}]。候选仅按valid选择，test未用于调参。",
             "", "## 同口径主表", "",
             "| 模型 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |", "|---|---:|---:|---:|---:|"]
    for name in models:
        lines.append(f"| {MODEL_LABELS[name]} | " + " | ".join(
            _format_score(all_summaries[name]["all"][metric]) for metric in METRICS) + " |")
    baseline_lo, baseline_hi = baseline_ci["ci95"]
    baseline_conclusion = ("区间包含0，当前结果不足以确认总体优于GRU4Rec。"
                           if baseline_lo <= 0 <= baseline_hi else
                           "该区间描述当前已训练模型条件下的用户差异，不代表跨训练或跨数据集的保证。")
    lines += ["", f"相对GRU4Rec，候选的NDCG@10均值差为{baseline_ci['delta']:+.6f}，"
              f"用户配对bootstrap 95%区间为[{baseline_lo:+.6f}, {baseline_hi:+.6f}]。"
              + baseline_conclusion + "[原始统计](summaries/gru4rec_comparison.json)"]
    lines += ["", "![总体与长尾NDCG对比](figures/test_comparison.png)", "",
              "误差线为三个训练种子的总体标准差，散点为各个种子；完整分组的四项指标见[test_results.csv](summaries/test_results.csv)。",
              "", "## 三种子与分组", "",
              "| 生成方案 | seed42 NDCG@10 | seed43 | seed44 |", "|---|---:|---:|---:|"]
    for name in variants:
        lines.append(f"| {MODEL_LABELS[name]} | " + " | ".join(
            f"{records[name, seed]['metrics']['all']['ndcg@10']:.6f}" for seed in (42, 43, 44)) + " |")
    lines += ["", "| 目标物品分组 | n | SASRec | GRU4Rec | Text | " + selected.replace("T5 · ", "") + " |",
              "|---|---:|---:|---:|---:|---:|"]
    for group in GROUP_LABELS:
        lines.append(f"| {GROUP_LABELS[group]} | {all_summaries['text'][group]['n']} | " + " | ".join(
            _format_score(all_summaries[name][group]["ndcg@10"]) for name in models) + " |")
    comparisons = [all_summaries[selection["selected"]][group]["ndcg@10"]["mean"]
                   - all_summaries["gru4rec"][group]["ndcg@10"]["mean"]
                   for group in ("head_top20", "tail_bottom80", "few_shot_le20")]
    lines += ["", "分组表为NDCG@10。分组由目标物品的train频次定义；少样本组包含在长尾组内，不应相加。",
              "", f"候选相对GRU4Rec的均值差：头部{comparisons[0]:+.4f}，长尾{comparisons[1]:+.4f}，"
              f"少样本{comparisons[2]:+.4f}。总体比较应同时看分组结果。",
              "", "## 四组表示的valid消融（seed42）", "",
              "| 表示 | valid NDCG@10 |", "|---|---:|"]
    for name in VARIANTS:
        lines.append(f"| {MODEL_LABELS[name].replace('T5 · ', '')} | {selection['scores'][name]:.6f} |")
    lines += ["", f"按valid冻结{selected}，随后补齐它与Text的三个训练种子。"
              "Adaptive未在该筛选中超过Fixed；Fixed-Match只匹配平均协同强度。"
              "单seed结果不足以确定频次分配策略的稳定收益或失败原因。",
              "", "## 结论与适用边界", "",
              "- 协同信号在量化前进入物品ID；Text与候选使用相同的模型、训练预算和解码设置。"
              "前缀协同相似度的变化提供描述性证据，不能单独证明收益机制。",
              "- 主指标同时报告候选相对Text与GRU4Rec的用户配对区间。"
              "数据、历史与评价口径对齐，不代表不同架构的参数量和计算预算完全相同。",
              "- 配对bootstrap执行10,000次用户重采样，固定当前三个已训练模型。"
              "它描述用户抽样不确定性，训练波动另看种子标准差。",
              f"- 仅验证一个数据集、三个推荐模型训练种子；量化种子固定42。少样本test仅{all_summaries['text']['few_shot_le20']['n']}人，严格冷启动n=0，记为N/A。",
              "- 固定训练预算不等于充分收敛，需结合[训练轨迹](../05-formal-training/README.md)和[量化质量](../03-semantic-ids/README.md)判断。"
              "旧30k与新60k还同时改变学习率轨迹和表示设置，未单独归因训练步数。",
              "- 搜索诊断使用旧checkpoint与固定512名valid用户；exact是模型概率排序参考，不是NDCG上界。"
              "离线指标与批处理耗时不能替代生产系统的在线效果、延迟与成本验证。",
              "", "## 历史配置参考", "",
              "Week 1–6的h200基线NDCG@10为SASRec 0.1637±0.0034、GRU4Rec 0.1882±0.0009；"
              "旧h20生成模型为0.1180±0.0092。历史配置的选模口径与本周存在差异，"
              "该结果只作配置效果参考，不用于推断架构优劣。"]
    out = folder / "main_results.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{out}: all twelve test results verified from full predictions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("curves", "diagnostics", "select", "summarize"))
    args = parser.parse_args()
    {"curves": curves, "diagnostics": diagnostics,
     "select": select, "summarize": summarize}[args.action]()


if __name__ == "__main__":
    main()
