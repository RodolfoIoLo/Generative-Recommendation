from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


METRICS = ["recall@5", "recall@10", "ndcg@5", "ndcg@10"]
GROUPS = ["all", "few_shot_le20", "tail_bottom80", "head_top20"]
SEEDS = {42, 43, 44}


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
    records = []
    for path in sorted(result_dir.glob("pilot_lr*_history.jsonl")):
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
        reason = "difference < 0.002; selected the more conservative learning rate"
    else:
        selected = max(scores, key=scores.get)
        reason = "selected the higher final-step validation NDCG@10"
    lines = [
        "# Learning-rate pilot",
        "",
        "| lr | final step | valid NDCG@10 |",
        "|---:|---:|---:|",
    ]
    lines.extend(f"| {lr} | {step} | {score:.6f} |" for lr, step, score in records)
    lines.extend(["", f"Selected: `{selected}` — {reason}.", ""])
    (result_dir / "pilot_summary.md").write_text("\n".join(lines), encoding="utf-8")
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
    generated = {}
    for split in ("valid", "test"):
        for mode in ("unconstrained", "constrained"):
            paths = sorted(result_dir.glob(f"generative_{split}_{mode}_seed*_metrics.json"))
            assert {seed_from_name(path) for path in paths} == SEEDS, (split, mode, paths)
            generated[(split, mode)] = [load_json(path) for path in paths]
            for record in generated[(split, mode)]:
                verify_metric_file(record)
                if mode == "constrained":
                    assert record["diagnostics"]["catalog_invalid_rate"] == 0.0
                    assert record["diagnostics"]["seen_repeat_rate"] == 0.0

    for mode in ("unconstrained", "constrained"):
        paths = sorted(result_dir.glob(f"generative_test_{mode}_seed*_predictions.jsonl"))
        assert {seed_from_name(path) for path in paths} == SEEDS
        for path in paths:
            verify_prediction_file(path, constrained=(mode == "constrained"))

    baselines = {}
    any_baseline = any(result_dir.glob("baseline_*_test_seed*_metrics.json"))
    if any_baseline:
        for model in ("sasrec", "gru4rec"):
            paths = sorted(result_dir.glob(f"baseline_{model}_test_seed*_metrics.json"))
            assert {seed_from_name(path) for path in paths} == SEEDS, (model, paths)
            baselines[model] = [load_json(path) for path in paths]
            for record in baselines[model]:
                verify_metric_file(record)
    return generated, baselines


def make_decode_samples(result_dir):
    unconstrained = {
        row["user"]: row
        for row in load_jsonl(result_dir / "generative_test_unconstrained_seed42_predictions.jsonl")
    }
    constrained = load_jsonl(
        result_dir / "generative_test_constrained_seed42_predictions.jsonl"
    )
    indices = np.linspace(0, len(constrained) - 1, 20, dtype=int)
    lines = [
        "# Decode samples (seed 42, fixed 20 users)", "",
        "`null` means an illegal, seen, or duplicate raw beam slot.", "",
        "| user | target | unconstrained Top-10 | constrained Top-10 |", "|---:|---:|---|---|",
    ]
    for index in indices:
        c = constrained[int(index)]
        u = unconstrained[c["user"]]
        lines.append(
            f"| {c['user']} | {c['target_item']} | `{u['ranked_items']}` | `{c['ranked_items']}` |"
        )
    (result_dir / "decode_samples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def series_table(title, model_records, group="all"):
    lines = [
        f"## {title}", "",
        "| Model | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |",
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
    lines = ["# Week 5–6: generative recommendation results", ""]
    lines += series_table("Main experiment (test, mean±population std)", models)

    lines += [
        "## Constrained decoding", "",
        "| Mode | Catalog invalid | Seen repeat | Short list |",
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
        "## Grouped test evaluation", "",
        "Strict cold-start: **N/A (n=0)**. Few-shot is a proxy, not strict cold-start.", "",
        "| Model | Group | n | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model, records in models:
        for group in GROUPS:
            n_values = {row["metrics"][group]["n"] for row in records}
            assert len(n_values) == 1
            n = n_values.pop()
            cells = [format_cell([row["metrics"][group][metric] for row in records]) for metric in METRICS]
            lines.append(f"| {model} | {group} | {n} | " + " | ".join(cells) + " |")
    lines += [""]
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
    output = Path(args.output) if args.output else result_dir / "main_results.md"
    output.write_text(report + "\n", encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()