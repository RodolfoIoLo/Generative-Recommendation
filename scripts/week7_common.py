"""Week 7 shared utilities. Importing this module needs only NumPy."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

PROTOCOL = "week7-v5.1"
ROOT = Path("results/week7-improvement")
WORK = Path("work/week7")
VARIANTS = ("text", "fixed", "fixed_match", "adaptive")
METRICS = ("recall@5", "recall@10", "ndcg@5", "ndcg@10")
ARTIFACT_FILES = (
    "semantic_ids.json", "train.semantic.jsonl", "valid.semantic.jsonl",
    "test.semantic.jsonl", "token_vocab.json", "meta.json",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                               sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False,
                                   allow_nan=False) + "\n")


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False,
                                allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                    ensure_ascii=False).encode()).hexdigest()


def artifact_hash(folder):
    return object_hash({name: sha256(Path(folder) / name)
                        for name in ARTIFACT_FILES})


def load_config(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def unit_rows(array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("embedding must be a finite 2-D array")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("zero-norm embedding row")
    return array / norms


def fusion_weights(freq, quantile=0.8):
    freq = np.asarray(freq, dtype=np.float64)
    q = float(np.quantile(freq, quantile))
    if q <= 0 or np.any(freq < 0):
        raise ValueError("invalid train frequencies")
    adaptive = np.clip(np.log1p(freq) / np.log1p(q), 0, 1)
    # Match average collaborative squared norm after joint normalization.
    mean_energy = float(np.mean(adaptive ** 2 / (1 + adaptive ** 2)))
    matched = math.sqrt(mean_energy / (1 - mean_energy))
    return {
        "text": np.zeros_like(freq),
        "fixed": np.ones_like(freq),
        "fixed_match": np.full_like(freq, matched),
        "adaptive": adaptive,
    }, {"q80": q, "matched_weight": matched,
        "adaptive_mean_collab_energy": mean_energy}


def fuse(text, collab, weights):
    text, collab = unit_rows(text), unit_rows(collab)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
    if len(text) != len(collab) or len(text) != len(weights):
        raise ValueError("item order/shape mismatch")
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("invalid fusion weights")
    return unit_rows(np.concatenate([text, weights * collab], axis=1))


def train_items(frozen):
    """Only read train.semantic.jsonl and its bijective item dictionary."""
    frozen = Path(frozen)
    ids = read_json(frozen / "semantic_ids.json")
    if set(ids) != {str(i) for i in range(len(ids))}:
        raise ValueError("item IDs must be contiguous 0..N-1")
    inverse = {tuple(code): int(item) for item, code in ids.items()}
    if len(inverse) != len(ids):
        raise ValueError("semantic IDs are not unique")
    rows = read_jsonl(frozen / "train.semantic.jsonl")
    users = [int(row["user"]) for row in rows]
    if len(users) != len(set(users)):
        raise ValueError("duplicate train user")
    return [{"user": int(row["user"]),
             "items": [inverse[tuple(code)] for code in row["items"]]}
            for row in sorted(rows, key=lambda row: int(row["user"]))], ids


def ranking_metrics(ranked, target):
    result = {}
    for k in (5, 10):
        rank = ranked[:k].index(target) + 1 if target in ranked[:k] else None
        result[f"recall@{k}"] = float(rank is not None)
        result[f"ndcg@{k}"] = 0.0 if rank is None else 1 / math.log2(rank + 1)
    return result


def scheduler_ratio(step, warmup, total, floor=0.05):
    if not 0 <= floor <= 1 or not 0 < warmup < total:
        raise ValueError("invalid scheduler")
    if step < warmup:
        return step / max(1, warmup)
    progress = min(1.0, (step - warmup) / (total - warmup))
    return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2


def require_test_unlock(variant, seed):
    selection = read_json(ROOT / "selection.json")
    if variant not in ("text", selection["selected"]):
        raise ValueError("test is restricted to the frozen final variants")
    for name in ("text", selection["selected"]):
        for run_seed in (42, 43, 44):
            meta = read_json(ROOT / "05-formal-training" /
                             f"{name}_s{run_seed}_best.json")
            if not meta["finished"] or meta["smoke"]:
                raise ValueError("finish all six formal runs before test")
            if meta["artifact_hash"] != selection["artifact_hashes"][name]:
                raise ValueError("artifacts changed after selection")
    return selection


def configure_gpu(gpu):
    """Called before importing torch. An idle observation is not a reservation."""
    if gpu is not None:
        if int(gpu) < 0:
            raise ValueError("physical GPU index must be nonnegative")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def environment():
    import importlib.metadata
    import platform
    names = ("numpy", "scipy", "scikit-learn", "torch", "transformers", "recbole")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "packages": versions}