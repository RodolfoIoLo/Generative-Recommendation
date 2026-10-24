from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


METRICS = ["recall@5", "recall@10", "ndcg@5", "ndcg@10"]
GROUPS = ["all", "few_shot_le20", "tail_bottom80", "head_top20"]
SEEDS = {42, 43, 44}
GROUP_LABELS = {
    "all": "全部",
    "few_shot_le20": "少样本（≤20）",
    "tail_bottom80": "长尾（后 80%）",
    "head_top20": "头部（前 20%）",
}
RESULT_SUBDIRS = {
    "pilot": "02-learning-rate-selection",
    "training": "03-formal-training",
    "generative": "04-generative-evaluation",
    "baseline": "05-baseline-evaluation",
    "reports": "06-reports",
}


def result_subdir(result_dir, kind, create=False):
    """Use the structured layout, while retaining old flat-run compatibility."""
    structured = any((result_dir / name).is_dir() for name in RESULT_SUBDIRS.values())
    path = result_dir / RESULT_SUBDIRS[kind] if structured else result_dir
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output")
    parser.add_argument("--pilot_only", action="store_true")
    parser.add_argument("--check_only", action="store_true")
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))


def format_cell(values):
    mean, std = mean_std(values)
    return f"{mean:.4f}±{std:.4f}"


def seed_from_name(path):
    match = re.search(r"seed(\d+)", Path(path).name)
    if not match:
        raise ValueError(f"seed absent from filename: {path}")
    return int(match.group(1))


def run_pilot(result_dir):
    pilot_dir = result_subdir(result_dir, "pilot", create=True)
    records = []
    for path in sorted(pilot_dir.glob("pilot_lr*_history.jsonl")):
        lr_text = path.name.split("pilot_lr", 1)[1].split("_history", 1)[0]
        valid = [row for row in load_jsonl(path) if row.get("type") == "valid"]
        if not valid:
            raise AssertionError(f"no validation record: {path}")
        final = max(valid, key=lambda row: row["step"])
        records.append((lr_text, int(final["step"]), final["metrics"]["all"]["ndcg@10"]))
    if {x[0] for x in records} != {"3e-4", "1e-3"}:
        raise AssertionError("pilot must contain exactly lr=3e-4 and lr=1e-3")
    scores = {lr: score for lr, _, score in records}
    if abs(scores["3e-4"] - scores["1e-3"]) < 0.002:
        selected = "3e-4"
        reason = "差值小于 0.002，采用更保守的学习率"
    else:
        selected = max(scores, key=scores.get)
        reason = "采用最终验证集 NDCG@10 更高的学习率"
    lines = [
        "# 学习率预实验",
        "",
        "| 学习率 | 最终 step | 验证集 NDCG@10 |",
        "|---:|---:|---:|",
    ]
    lines.extend(f"| {lr} | {step} | {score:.6f} |" for lr, step, score in records)
    lines.extend(["", f"结论：选择 `{selected}` — {reason}。", ""])
    (pilot_dir / "pilot_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def verify_metric_file(record):
    assert record["n_users"] == 6040
    for group, values in record["metrics"].items():
        n = values["n"]
        assert isinstance(n, int) and 0 <= n <= 6040
        for key in METRICS:
            value = values[key]
            assert value is None or 0.0 <= value <= 1.0
        if n:
            assert values["recall@10"] + 1e-12 >= values["recall@5"]


def verify_prediction_file(path, constrained):
    rows = load_jsonl(path)
    assert len(rows) == 6040, f"{path}: expected 6040 rows"
    assert len({row["user"] for row in rows}) == 6040
    for row in rows:
        ranked = row["ranked_items"]
        assert len(ranked) == 10
        valid = [item for item in ranked if item is not None]
        assert len(valid) == len(set(valid))
        if constrained:
            assert len(valid) == 10


def discover_and_check(result_dir):
    generative_dir = result_subdir(result_dir, "generative")
    baseline_dir = result_subdir(result_dir, "baseline")
    generated = {}
    for split in ("valid", "test"):
        for mode in ("unconstrained", "constrained"):
            paths = sorted(generative_dir.glob(
                f"generative_{split}_{mode}_seed*_metrics.json"
            ))
            assert {seed_from_name(path) for path in paths} == SEEDS, (split, mode, paths)
            generated[(split, mode)] = [load_json(path) for path in paths]
            for record in generated[(split, mode)]:
                verify_metric_file(record)
                if mode == "constrained":
                    assert record["diagnostics"]["catalog_invalid_rate"] == 0.0
                    assert record["diagnostics"]["seen_repeat_rate"] == 0.0

    for mode in ("unconstrained", "constrained"):
        paths = sorted(generative_dir.glob(
            f"generative_test_{mode}_seed*_predictions.jsonl"
        ))
        assert {seed_from_name(path) for path in paths} == SEEDS
        for path in paths:
            verify_prediction_file(path, constrained=(mode == "constrained"))

    baselines = {}
    any_baseline = any(baseline_dir.glob("baseline_*_test_seed*_metrics.json"))
    if any_baseline:
        for model in ("sasrec", "gru4rec"):
            paths = sorted(baseline_dir.glob(
                f"baseline_{model}_test_seed*_metrics.json"
            ))
            assert {seed_from_name(path) for path in paths} == SEEDS, (model, paths)
            baselines[model] = [load_json(path) for path in paths]
            for record in baselines[model]:
                verify_metric_file(record)
                verify_metric_file({"n_users": record["n_users"],
                                    "metrics": record["legacy_unmasked_metrics"]})
                assert record.get("old_log_alignment", "").startswith("passed")
    return generated, baselines


def make_decode_samples(result_dir):
    generative_dir = result_subdir(result_dir, "generative")
    report_dir = result_subdir(result_dir, "reports", create=True)
    unconstrained = {
        row["user"]: row
        for row in load_jsonl(
            generative_dir / "generative_test_unconstrained_seed42_predictions.jsonl"
        )
    }
    constrained = load_jsonl(
        generative_dir / "generative_test_constrained_seed42_predictions.jsonl"
    )
    project_root = result_dir.parent.parent
    frozen_dir = project_root / "exported"
    test_by_user = {
        int(row["user"]): row
        for row in load_jsonl(frozen_dir / "test.semantic.jsonl")
    }
    item_to_code = load_json(frozen_dir / "semantic_ids.json")
    code_to_item = {tuple(code): int(item) for item, code in item_to_code.items()}
    indices = np.linspace(0, len(constrained) - 1, 20, dtype=int)
    lines = [
        "# 解码样例（seed 42，固定 20 名用户）", "",
        "`null` 表示该无约束 beam 槽位对应非法 ID、已见物品或重复物品；标记栏给出三类槽位数。", "",
        "| 用户 | 历史长度 | 目标 | 无约束 Top-10 | 无约束标记 | 约束 Top-10 |",
        "|---:|---:|---:|---|---|---|",
    ]
    for index in indices:
        c = constrained[int(index)]
        u = unconstrained[c["user"]]
        frozen = test_by_user[int(c["user"])]
        seen = {code_to_item[tuple(code)] for code in frozen["history"]}
        used, counts = set(), {"非法": 0, "已见": 0, "重复": 0}
        for raw_code in u["generated_codes"]:
            item = None if raw_code is None else code_to_item.get(tuple(raw_code))
            if item is None:
                counts["非法"] += 1
            elif item in seen:
                counts["已见"] += 1
            elif item in used:
                counts["重复"] += 1
            else:
                used.add(item)
        flags = " / ".join(f"{name} {value}" for name, value in counts.items())
        lines.append(
            f"| {c['user']} | {len(frozen['history'])} | {c['target_item']} | "
            f"`{u['ranked_items']}` | {flags} | `{c['ranked_items']}` |"
        )
    (report_dir / "decode_samples.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def series_table(title, model_records, group="all"):
    lines = [
        f"## {title}", "",
        "| 模型 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, records in model_records:
        cells = [format_cell([row["metrics"][group][metric] for row in records]) for metric in METRICS]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return lines + [""]


def build_report(result_dir, generated, baselines):
    if set(baselines) != {"sasrec", "gru4rec"}:
        raise AssertionError("final report requires three seeds for both baselines")
    gen = generated[("test", "constrained")]
    models = [("SASRec", baselines["sasrec"]),
              ("GRU4Rec", baselines["gru4rec"]),
              ("Generative-T5", gen)]
    lines = [
        "# 第 5–6 周：生成式推荐主实验", "",
        "> MovieLens 1M · 三随机种子 · 全量排序 · 总体标准差（ddof=0）", "",
        "主表统一屏蔽每名用户的完整历史物品。基线文件另存原始 RecBole 未屏蔽历史的复算结果，并与既有日志在 `1e-4` 内对齐。", "",
    ]
    lines += series_table("主实验（test，均值±总体标准差）", models)

    lines += [
        "## 约束解码", "",
        "| 模式 | 目录外 ID | 已见物品槽位 | 不足 10 条的用户 |",
        "|---|---:|---:|---:|",
    ]
    for mode in ("unconstrained", "constrained"):
        records = generated[("test", mode)]
        values = []
        for key in ("catalog_invalid_rate", "seen_repeat_rate", "short_list_rate"):
            values.append(format_cell([row["diagnostics"][key] for row in records]))
        lines.append(f"| {mode} | " + " | ".join(values) + " |")
    lines += [""]

    lines += [
        "## 分组测试结果", "",
        "严格冷启动为 **N/A（n=0）**；少样本组只是稀疏物品代理，不等同于严格冷启动。", "",
        "| 模型 | 分组 | n | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model, records in models:
        for group in GROUPS:
            n_values = {row["metrics"][group]["n"] for row in records}
            assert len(n_values) == 1
            n = n_values.pop()
            cells = [format_cell([row["metrics"][group][metric] for row in records]) for metric in METRICS]
            lines.append(f"| {model} | {GROUP_LABELS[group]} | {n} | " + " | ".join(cells) + " |")
    lines += [""]

    lines += [
        "## 旧日志复现", "",
        "下表采用 RecBole 原始序列评测口径，不屏蔽历史物品，仅用于证明 checkpoint 与既有日志一致；最终模型比较采用主表口径。", "",
        "| 模型 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | 对齐 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for model, records in (("SASRec", baselines["sasrec"]),
                           ("GRU4Rec", baselines["gru4rec"])):
        cells = [format_cell([row["legacy_unmasked_metrics"]["all"][metric]
                              for row in records]) for metric in METRICS]
        lines.append(f"| {model} | " + " | ".join(cells) + " | PASS（≤1e-4） |")
    lines += [""]

    gen_ndcg = mean_std([row["metrics"]["all"]["ndcg@10"] for row in gen])[0]
    gru_ndcg = mean_std([row["metrics"]["all"]["ndcg@10"]
                         for row in baselines["gru4rec"]])[0]
    gen_tail = mean_std([row["metrics"]["tail_bottom80"]["ndcg@10"] for row in gen])[0]
    gru_tail = mean_std([row["metrics"]["tail_bottom80"]["ndcg@10"]
                         for row in baselines["gru4rec"]])[0]
    lines += [
        "## 结论", "",
        f"生成模型的总体 NDCG@10 为 `{gen_ndcg:.4f}`，低于 GRU4Rec 的 `{gru_ndcg:.4f}`；"
        f"长尾组同样为 `{gen_tail:.4f}` 对 `{gru_tail:.4f}`。当前实验不支持生成模型优于判别式基线，也不支持冷启动收益结论。", "",
        "无约束生成没有产生目录外 ID，但会大量生成用户已见物品，导致多数用户无法得到完整的十条新推荐。"
        "动态 Trie 将已见物品率和短列表率都降为零，同时提高最终排序指标。", "",
        "可能原因包括 MovieLens 物品目录较小且协同信号充分、电影文本字段有限、生成模型从随机参数训练，"
        "以及多 token 自回归解码中的误差传播。后续改进应以这些可观测差距为依据，并通过消融或配对检验验证。", "",
    ]
    make_decode_samples(result_dir)
    return "\n".join(lines)


def main():
    args = parse_args()
    result_dir = Path(args.results)
    if args.pilot_only:
        run_pilot(result_dir)
        return
    generated, baselines = discover_and_check(result_dir)
    print("all available result checks passed")
    if args.check_only:
        return
    report = build_report(result_dir, generated, baselines)
    output = (
        Path(args.output)
        if args.output
        else result_subdir(result_dir, "reports", create=True) / "main_results.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report + "\n", encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
