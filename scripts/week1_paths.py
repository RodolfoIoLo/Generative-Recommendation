"""Result locations shared by the five experiment stages.

Historical JSON records retain their original paths. Resolve those paths here
without rewriting frozen metadata or changing experiment identities.
"""
from pathlib import Path
import re


def result_path(value):
    path = Path(value)
    parts = path.parts
    try:
        index = parts.index("results")
        week = parts[index + 1]
    except (ValueError, IndexError):
        return path
    relative = parts[index + 2:]
    if not relative or not path.suffix:
        return path
    root, name = Path(*parts[:index + 2]), path.name
    stage = relative[0]
    if week == "week2-sasrec-baseline" and name.endswith(".log"):
        model = "sasrec" if name.startswith("sasrec") else "gru4rec"
        kind = "evaluation-logs" if "patch" in name else "training-logs"
        return root / kind / model / name
    if week == "week3-4-semantic-id":
        if stage == "01-quantization" and name.endswith((".json", ".log")):
            return root / stage / ("identifiers" if name.endswith(".json") else "logs") / name
        if stage == "03-grid-search" and name.endswith((".json", ".log")) and name != "grid_search.log":
            model = "kmeans" if "kmeans" in name else "rqvae"
            return root / stage / ("candidates" if name.endswith(".json") else "logs") / model / name
        if stage == "02-codebook-analysis" and name.endswith(".png"):
            model = "kmeans" if "kmeans" in name else "rqvae"
            return root / stage / "figures" / model / name
    if week == "week5-6-generative":
        if name.startswith("pilot_lr"):
            lr = name.split("pilot_lr", 1)[1].split("_", 1)[0].removesuffix(".log")
            return root / "02-learning-rate-selection" / ("lr" + lr) / name
        if name.startswith(("smoke", "overfit256")):
            group = "smoke" if name.startswith("smoke") else "overfit"
            return root / "01-pipeline-validation" / group / name
        if name in ("audit.log", "stage1_static_checks.log", "stage0-3_report.md"):
            return root / "01-pipeline-validation" / "audit" / name
        if name == "prepare_report.md":
            return root / "01-pipeline-validation" / name
        if name == "pilot_summary.md":
            return root / "02-learning-rate-selection" / name
        if name == "baseline_eval.log":
            return root / "05-baseline-evaluation" / name
        if name in ("main_results.md", "decode_samples.md"):
            return root / "06-reports" / name
        match = re.match(r"(?:t5_sid_seed|train_seed)(\d+)", name)
        if match:
            return root / "03-formal-training" / ("seed" + match[1]) / name
        match = re.match(r"generative_(valid|test)_(constrained|unconstrained)_", name)
        if match:
            return root / "04-generative-evaluation" / match[1] / match[2] / name
        match = re.match(r"baseline_(sasrec|gru4rec)_", name)
        if match:
            return root / "05-baseline-evaluation" / match[1] / name
    if week == "week7-improvement":
        if name.startswith("smoke_"):
            return root / "01-diagnostics" / "cpu-smoke" / name
        if name in ("environment.json", "input_sha256.txt", "reused_code_sha256.txt"):
            return root / "01-diagnostics" / "environment" / name
        if name in ("training_curves.csv", "training_curves.png"):
            return root / "01-diagnostics" / "training-curves" / name
        if name.startswith("historical_30k_s42_valid_") and name.endswith(".json"):
            kind = "samples" if name.endswith("_samples.json") else "metrics"
            return root / "01-diagnostics" / "search" / kind / name
        if name == "search_diagnosis.md":
            return root / "01-diagnostics" / "search" / name
        if stage == "03-semantic-ids" and len(relative) >= 3:
            variant = relative[1]
            if name in ("semantic_ids.json", "meta.json"):
                return root / stage / variant / "identifiers" / name
            if name in ("quantizer_quality.json", "codebook_stats.json", "prefix_quality.json"):
                return root / stage / variant / "quality" / name
            if name == "rqvae.log":
                return root / stage / variant / "training" / name
        if stage == "04-screening" and name.endswith(".log"):
            return root / stage / "training-logs" / name
        if stage == "05-formal-training":
            match = re.match(r"(text|fixed_match|fixed|adaptive|historical)(?:_s|-s)(\d+)", name)
            if match:
                return root / stage / "generative" / match[1] / ("seed" + match[2]) / name
            match = re.match(r"(sasrec|gru4rec)_h20_s(\d+)", name)
            if match:
                return root / stage / "baselines" / match[1] / ("seed" + match[2]) / name
        if stage == "06-evaluation":
            match = re.match(r"(sasrec|gru4rec)_h20_s(\d+)_", name)
            if match:
                return root / stage / "baselines" / match[1] / ("seed" + match[2]) / name
            match = re.match(r"(text|fixed_match|fixed|adaptive|historical)_s(\d+)_", name)
            if match:
                return root / stage / "generative" / match[1] / ("seed" + match[2]) / name
            if name in ("summary.json", "baseline_h20_summary.json", "paired_bootstrap.json",
                        "gru4rec_comparison.json", "test_results.csv"):
                return root / stage / "summaries" / name
            if name == "test_comparison.png":
                return root / stage / "figures" / name
    return path


def output_path(value):
    path = result_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
