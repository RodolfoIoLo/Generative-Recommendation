"""CPU preparation, configs, quantization, and artifact export."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from week7_common import (
    ARTIFACT_FILES, PROTOCOL, ROOT, WORK, VARIANTS, artifact_hash,
    configure_gpu, environment, fuse, fusion_weights, read_json, read_jsonl, sha256,
    train_items, unit_rows, write_json, write_jsonl,
)


def initialize():
    import yaml
    for folder in ("01-diagnostics", "02-embeddings", "03-semantic-ids",
                   "04-screening", "05-formal-training", "06-evaluation",
                   "07-release"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    Path("config/week7").mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(Path("config/generative_ml1m.yaml").read_text())
    base["paths"].update(
        checkpoint_root="saved/week7/generative",
        result_root=str(ROOT),
    )
    base["train"].update(
        max_steps=60000, warmup_steps=1500, min_lr_ratio=0.05,
        lr=0.0003, eval_steps=5000, num_workers=0, patience=4,
        early_stop_after=30000, min_delta=0.001,
    )
    for name in (*VARIANTS, "historical"):
        cfg = copy.deepcopy(base)
        cfg["protocol_version"] = PROTOCOL
        cfg["variant"] = name
        cfg["paths"]["frozen_dir"] = (
            "exported" if name == "historical" else f"exported/week7/{name}"
        )
        path = Path(f"config/week7/generative_{name}.yaml")
        if path.exists():
            if yaml.safe_load(path.read_text()) != cfg:
                raise ValueError(f"refusing to overwrite changed config: {path}")
        else:
            path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    rq = yaml.safe_load(Path("config/rqvae_ml1m.yaml").read_text())
    rq.update(in_dim=448, seed=42, device="cpu")
    path = Path("config/week7/rqvae.yaml")
    if path.exists() and yaml.safe_load(path.read_text()) != rq:
        raise ValueError(f"refusing to overwrite changed config: {path}")
    path.write_text(yaml.safe_dump(rq, sort_keys=False), encoding="utf-8")
    write_json(ROOT / "01-diagnostics/environment.json", environment())
    print("config/week7: five generator configs + one RQ-VAE config")


def build_embeddings(frozen="exported", text_path="item_emb.npy"):
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD
    rows, ids = train_items(frozen)
    text = unit_rows(np.load(text_path))
    if text.shape != (3416, 384) or len(rows) != 6040 or len(ids) != 3416:
        raise ValueError("unexpected frozen MovieLens dimensions")
    pairs = [(u, item) for u, row in enumerate(rows)
             for item in set(row["items"])]
    u, i = np.asarray(pairs, dtype=np.int64).T
    matrix = csr_matrix((np.ones(len(pairs), dtype=np.float32), (u, i)),
                        shape=(len(rows), len(ids)))
    svd = TruncatedSVD(n_components=64, n_iter=10, random_state=42)
    collab = unit_rows(svd.fit_transform(matrix.T))
    # Counts use all observed train occurrences, not binary-matrix degrees.
    freq = np.bincount([item for row in rows for item in row["items"]],
                       minlength=len(ids))
    weights, weight_meta = fusion_weights(freq)
    np.save(WORK / "collab.npy", collab)
    for name, weight in weights.items():
        np.save(WORK / f"{name}.npy", fuse(text, collab, weight))
    write_json(WORK / "frequency.json", {str(i): int(x)
                                       for i, x in enumerate(freq)})
    write_json(ROOT / "02-embeddings/weights.json",
               {name: weight.tolist() for name, weight in weights.items()})
    write_json(ROOT / "02-embeddings/meta.json", {
        "protocol_version": PROTOCOL, "source": "train_only",
        "train_sha256": sha256(Path(frozen) / "train.semantic.jsonl"),
        "item_map_sha256": sha256(Path(frozen) / "semantic_ids.json"),
        "text_sha256": sha256(text_path), "collab_sha256": sha256(WORK / "collab.npy"),
        "shape": list(collab.shape), "svd_seed": 42, "svd_n_iter": 10,
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "weight_definition": weight_meta,
        "variants": {name: {"sha256": sha256(WORK / f"{name}.npy"),
                            "shape": [3416, 448]}
                     for name in VARIANTS},
    })
    neighbors = []
    for item in np.linspace(0, len(ids) - 1, 20, dtype=int):
        score = collab @ collab[item]
        score[item] = -np.inf
        order = np.argsort(-score, kind="stable")[:10]
        neighbors.append({"item": int(item), "neighbors": order.tolist(),
                          "cosines": score[order].tolist()})
    write_json(ROOT / "02-embeddings/neighbors.json", neighbors)
    print("four arrays: (3416,448); collaborative array: (3416,64)")


def export_variant(name):
    from export_semantic import build_vocab
    target = Path(f"exported/week7/{name}")
    target.mkdir(parents=True, exist_ok=True)
    ids = read_json(WORK / f"semantic_ids_{name}.json")
    old = read_json("exported/semantic_ids.json")
    inverse = {tuple(code): item for item, code in old.items()}
    if len(ids) != len(old) or len({tuple(x) for x in ids.values()}) != len(ids):
        raise ValueError("incomplete or duplicate semantic IDs")
    if any(len(code) != 4 or any(not 0 <= x < 256 for x in code)
           for code in ids.values()):
        raise ValueError("bad 4-position code")
    write_json(target / "semantic_ids.json", ids)
    for split in ("train", "valid", "test"):
        converted = []
        for row in read_jsonl(Path("exported") / f"{split}.semantic.jsonl"):
            new = copy.deepcopy(row)
            for key in ("items", "history"):
                if key in new:
                    new[key] = [ids[inverse[tuple(code)]] for code in new[key]]
            if "target" in new:
                new["target"] = ids[inverse[tuple(new["target"])]]
            converted.append(new)
        if len(converted) != 6040:
            raise ValueError("split user count changed")
        write_jsonl(target / f"{split}.semantic.jsonl", converted)
    vocab = read_json("exported/token_vocab.json")
    token2id, size = build_vocab(4, 256)
    vocab.update(token2id=token2id, vocab_size=size, semantic_id_length=4)
    write_json(target / "token_vocab.json", vocab)
    write_json(target / "meta.json", {
        "protocol_version": PROTOCOL, "variant": name,
        "n_items": len(ids), "embedding_dim": 448,
        "embedding_sha256": sha256(WORK / f"{name}.npy"),
        "quantizer_sha256": sha256(f"saved/week7/quantizers/{name}.pt"),
        "feature_meta_sha256": sha256(ROOT / "02-embeddings/meta.json"),
        "source_split_sha256": {
            split: sha256(Path("exported") / f"{split}.semantic.jsonl")
            for split in ("train", "valid", "test")
        },
    })
    write_json(ROOT / f"03-semantic-ids/{name}/manifest.json",
               {"artifact_hash": artifact_hash(target),
                "files": {file: sha256(target / file) for file in ARTIFACT_FILES}})


def tokenize(name):
    out = ROOT / "03-semantic-ids" / name
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(f"saved/week7/quantizers/{name}.pt")
    if checkpoint.exists():
        raise ValueError("quantizer already exists; do not silently retrain a frozen variant")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "scripts/build_semantic_id.py",
        "--config", "config/week7/rqvae.yaml",
        "--emb", str(WORK / f"{name}.npy"),
        "--method", "rqvae", "--in_dim", "448", "--device", "cpu",
        "--checkpoint", str(checkpoint),
        "--out", str(WORK / f"semantic_ids_{name}.json"),
    ]
    with (out / "rqvae.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace")
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        if process.wait():
            raise RuntimeError("RQ-VAE failed; inspect rqvae.log")
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if min(state["best_utilization"]) < 231 or state["best_r2"] <= 0:
        raise ValueError("quantizer failed utilization or reconstruction gate")
    export_variant(name)
    for command in (
        [sys.executable, "scripts/analyze_codebook.py", "--rqvae",
         str(WORK / f"semantic_ids_{name}.json"), "--outdir", str(out)],
    ):
        subprocess.run(command, check=True)
    write_json(out / "quantizer_quality.json",
               {key: state[key] for key in (
                   "best_epoch", "best_r2", "best_utilization", "best_perplexity")})
    # Existing prefix utility reads item text; this quantitative audit needs no text path.
    code_dict = read_json(WORK / f"semantic_ids_{name}.json")
    codes = np.array([code_dict[str(i)] for i in range(3416)])
    from check_prefix_consistency import group_mean_cos
    prefix = []
    for source, embedding in (("text", unit_rows(np.load("item_emb.npy"))),
                              ("collab", np.load(WORK / "collab.npy"))):
        rng = np.random.default_rng(42)
        left = rng.integers(0, len(embedding), 50000)
        right = rng.integers(0, len(embedding), 50000)
        valid = left != right
        random_mean = float((embedding[left[valid]] *
                             embedding[right[valid]]).sum(1).mean())
        for depth in (1, 2, 3):
            values, n_groups = group_mean_cos(codes, embedding, depth)
            prefix.append({
                "source": source, "depth": depth, "groups": n_groups,
                "pairs": len(values),
                "same_prefix_mean": float(values.mean()) if len(values) else None,
                "random_pair_mean": random_mean,
            })
    write_json(out / "prefix_quality.json", prefix)
    print(f"exported/week7/{name}: six artifacts exported and audited")


def main():
    configure_gpu(None)
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("init", "features", "tokenize"))
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    if args.action == "init":
        initialize()
    elif args.action == "features":
        build_embeddings()
    else:
        if not args.variant:
            parser.error("tokenize requires --variant")
        tokenize(args.variant)


if __name__ == "__main__":
    main()