"""Validate and package the public repository without rewriting documents."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = Path("results/week7-improvement")
METRICS = ("recall@5", "recall@10", "ndcg@5", "ndcg@10")
PUBLIC_DIRS = (".github", "config", "docs", "environment", "results", "scripts", "tests")
PUBLIC_FILES = (".gitignore", "README.md", "requirements.txt", "CITATION.cff", "LICENSE")
PRIVATE_PARTS = {"__pycache__", ".git", ".ipynb_checkpoints", "note", "work",
                 "saved", "exported", "log", "log_tensorboard"}
PRIVATE_SUFFIXES = {".pyc", ".pyo", ".pth", ".pt", ".safetensors", ".npy", ".npz", ".zip"}
SYNC_FILES = {"sync_manifest.json", "stage7.0-7.3_audit.json"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect_public_files(root):
    root = Path(root).resolve()
    paths = [root / name for name in PUBLIC_FILES]
    for folder in PUBLIC_DIRS:
        paths.extend(p for p in (root / folder).rglob("*") if p.is_file())
    selected = []
    for path in sorted(set(paths)):
        relative = path.relative_to(root)
        if (any(part in PRIVATE_PARTS for part in relative.parts)
                or path.suffix.lower() in PRIVATE_SUFFIXES or path.name in SYNC_FILES):
            continue
        if path.is_symlink() or root not in path.resolve().parents:
            raise ValueError(f"unexpected public symlink: {relative}")
        if not path.is_file():
            raise FileNotFoundError(relative)
        if path.stat().st_size > 32 * 1024**2:
            raise ValueError(f"review unexpectedly large public file: {relative}")
        selected.append(relative)
    return selected


def close(first, second):
    if not math.isclose(first, second, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"inconsistent summary: {first} != {second}")


def validate_results(root):
    root = Path(root)
    folder = root / RESULTS
    selection = read_json(folder / "selection.json")
    winner = selection["selected"]
    if winner not in ("fixed", "fixed_match", "adaptive") or selection["selection_split"] != "valid":
        raise ValueError("invalid frozen selection")
    required = ("README.md", "01-diagnostics/search_diagnosis.md",
                "01-diagnostics/training_curves.csv", "01-diagnostics/training_curves.png",
                "06-evaluation/README.md", "06-evaluation/main_results.md",
                "06-evaluation/limitations.md")
    for name in required:
        if not (folder / name).is_file():
            raise FileNotFoundError(folder / name)
    summaries = read_json(folder / "06-evaluation/summary.json")
    bootstrap = read_json(folder / "06-evaluation/paired_bootstrap.json")
    for name in ("text", winner):
        runs = []
        for seed in (42, 43, 44):
            trained = read_json(folder / "05-formal-training" / f"{name}_s{seed}_best.json")
            result = read_json(folder / "06-evaluation" / f"{name}_s{seed}_test_beam20_n6040.json")
            if (not trained["finished"] or trained["smoke"]
                    or trained["artifact_hash"] != selection["artifact_hashes"][name]
                    or result["artifact_hash"] != trained["artifact_hash"]
                    or result["variant"] != name or result["seed"] != seed
                    or result["split"] != "test" or result["n_users"] != 6040
                    or result["beam"] != 20 or not result["full_sort"] or not result["mask_full_history"]
                    or any(result["diagnostics"].values())):
                raise ValueError(f"invalid formal result: {name} seed{seed}")
            samples = read_json(folder / "06-evaluation" / f"{name}_s{seed}_test_beam20_n6040_samples.json")
            if len(samples) != 20:
                raise ValueError("missing public recommendation samples")
            runs.append(result)
        for group, summary in summaries[name].items():
            if any(r["metrics"][group]["n"] != summary["n"] for r in runs):
                raise ValueError("group sizes differ across seeds")
            for metric in METRICS:
                values = [r["metrics"][group][metric] for r in runs]
                if values[0] is None:
                    if summary[metric] is not None or any(v is not None for v in values):
                        raise ValueError("invalid empty-group summary")
                else:
                    close(summary[metric]["mean"], statistics.mean(values))
                    close(summary[metric]["std"], statistics.pstdev(values))
    for group, metrics in bootstrap.items():
        if metrics["ndcg@10"]["n_users"] != summaries["text"][group]["n"]:
            raise ValueError("bootstrap group size differs from evaluation")
        for metric, record in metrics.items():
            if record["n_users"]:
                close(record["delta"], summaries[winner][group][metric]["mean"]
                      - summaries["text"][group][metric]["mean"])
    baselines = read_json(folder / "06-evaluation/baseline_h20_summary.json")
    for model in ("sasrec", "gru4rec"):
        summary = next(row for row in baselines if row["model"] == model)
        runs = []
        for seed in (42, 43, 44):
            trained = read_json(folder / "05-formal-training" / f"{model}_h20_s{seed}.json")
            result = read_json(folder / "06-evaluation" / f"{model}_h20_s{seed}_test.json")
            if (result["split"] != "test" or result["n_users"] != 6040
                    or result["max_history"] != 20 or not result["mask_full_history"]
                    or not result["full_sort"] or result["seed"] != seed
                    or result["model"].lower() != model
                    or result["checkpoint_sha256"] != trained["checkpoint_sha256"]):
                raise ValueError(f"invalid baseline test result: {model} seed{seed}")
            runs.append(result)
            samples = read_json(folder / "06-evaluation" / f"{model}_h20_s{seed}_test_samples.json")
            if len(samples) != 20:
                raise ValueError("missing baseline recommendation samples")
        for metric in METRICS:
            values = [r["metrics"]["all"][metric] for r in runs]
            close(summary["metrics"][metric]["mean"], statistics.mean(values))
            close(summary["metrics"][metric]["std"], statistics.pstdev(values))
    return winner


def build_archive(root, output, files):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    expected = {}
    try:
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as bundle:
            for relative in files:
                content = (root / relative).read_bytes()
                expected[relative.as_posix()] = hashlib.sha256(content).hexdigest()
                bundle.writestr(relative.as_posix(), content)
        with zipfile.ZipFile(temp) as bundle:
            if set(bundle.namelist()) != set(expected) or bundle.testzip() is not None:
                raise ValueError("archive entries failed verification")
            for name, digest in expected.items():
                if hashlib.sha256(bundle.read(name)).hexdigest() != digest:
                    raise ValueError(f"archive content differs: {name}")
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT / "dist/week7-public.zip")
    # Accept the original taskbook's metadata flags without changing existing files.
    parser.add_argument("--author")
    parser.add_argument("--url")
    parser.add_argument("--license", choices=("MIT",))
    args = parser.parse_args()
    citation = (PROJECT / "CITATION.cff").read_text(encoding="utf-8")
    for value in (args.author, args.url):
        if value and value not in citation:
            raise ValueError("supplied metadata differs from CITATION.cff")
    if args.license and args.license not in (PROJECT / "LICENSE").read_text(encoding="utf-8"):
        raise ValueError("supplied license differs from LICENSE")
    winner = validate_results(PROJECT)
    files = collect_public_files(PROJECT)
    print(f"Public result validation PASS; selected={winner}; files={len(files)}")
    if not args.check_only:
        digest = build_archive(PROJECT, args.output, files)
        print(f"Verified archive: {args.output}; SHA-256={digest}")


if __name__ == "__main__":
    main()
