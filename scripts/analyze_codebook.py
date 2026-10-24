#!/usr/bin/env python
"""码本质量分析:逐层利用率 / 困惑度 / 冲突率 + 塌缩直方图.

用法:
    python scripts/analyze_codebook.py \
      --kmeans semantic_ids_kmeans.json --rqvae semantic_ids_rqvae.json \
      --outdir results/week3-4-semantic-id/02-codebook-analysis
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return np.array([d[str(i)] for i in range(len(d))], dtype=np.int64)


def stats(codes, codebook_size, has_dedup=True):
    """前 n_levels 位(不含消歧位)的逐层统计.

    has_dedup=True(默认)时丢掉最后一列(消歧码),因为消歧码没有码本含义.
    如果跑的是 --no_dedup 的备选方案,要传 has_dedup=False,否则会误删第 3 层.
    """
    n = len(codes)
    body = codes[:, :-1] if (has_dedup and codes.shape[1] > 1) else codes
    out = {"n_items": n, "n_levels": body.shape[1], "levels": []}
    out["n_unique_ids"] = len({tuple(r) for r in codes.tolist()})
    out["collision_rate_before_dedup"] = round(
        1 - len({tuple(r) for r in body.tolist()}) / n, 4)
    for l in range(body.shape[1]):
        col = body[:, l]
        u = int(len(np.unique(col)))
        p = np.bincount(col) / n
        p = p[p > 0]
        out["levels"].append({
            "level": l,
            "utilization": round(u / codebook_size, 4),
            "n_codes_used": u,
            "codebook_size": codebook_size,
            "perplexity": round(float(np.exp(-(p * np.log(p)).sum())), 2),
        })
    return out


def plot_hist(codes, outdir, tag, codebook_size, has_dedup=True):
    body = codes[:, :-1] if (has_dedup and codes.shape[1] > 1) else codes
    n = len(codes)
    for l in range(body.shape[1]):
        counts = np.bincount(body[:, l], minlength=codebook_size)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(counts, bins=40)
        ax.set_yscale("log")
        ax.set_xlabel("items assigned to a codeword")
        ax.set_ylabel("number of codewords (log)")
        ax.set_title(f"{tag} level {l}  (avg {n/codebook_size:.1f} items/codeword)")
        ax.grid(alpha=0.3)
        path = os.path.join(outdir, "figures", f"util_{tag}_level{l}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmeans", default=None)
    ap.add_argument("--rqvae", default=None)
    ap.add_argument("--codebook_size", type=int, default=256)
    ap.add_argument("--no_dedup", action="store_true",
                    help="语义 ID 末位不是消歧码时加这个(跑备选方案用)")
    ap.add_argument(
        "--outdir",
        default="results/week3-4-semantic-id/02-codebook-analysis",
    )
    args = ap.parse_args()

    has_dedup = not args.no_dedup
    result = {}
    for tag, path in [("kmeans", args.kmeans), ("rqvae", args.rqvae)]:
        if not path:
            continue
        codes = load(path)
        result[tag] = stats(codes, args.codebook_size, has_dedup)
        plot_hist(codes, args.outdir, tag, args.codebook_size, has_dedup)
        print(f"\n=== {tag} ===")
        for lv in result[tag]["levels"]:
            print(f"  level {lv['level']}: 利用率 "
                  f"{lv['n_codes_used']}/{lv['codebook_size']}"
                  f" = {lv['utilization']:.3f}|困惑度 {lv['perplexity']}")
        print(f"  消解前冲突率 {result[tag]['collision_rate_before_dedup']}"
              f"|消解后唯一 ID {result[tag]['n_unique_ids']}")

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "codebook_stats.json"),
              "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nsaved {args.outdir}/codebook_stats.json")


if __name__ == "__main__":
    main()
