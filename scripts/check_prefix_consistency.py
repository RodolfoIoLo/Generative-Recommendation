#!/usr/bin/env python
"""前缀语义一致性:同前缀物品对的平均余弦 vs 随机配对(bootstrap 95% 区间).

用法:
    python scripts/check_prefix_consistency.py \
      --emb item_emb.npy --ids semantic_ids_rqvae.json \
      --n_boot 1000 --outdir results/week3-4-semantic-id
"""
import argparse
import csv
import json
import os

import numpy as np


def load_ids(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return np.array([d[str(i)] for i in range(len(d))], dtype=np.int64)


def normed(emb):
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def group_mean_cos(ids, E, depth):
    """按前 depth 位分组,返回组内所有物品对的余弦,以及组数."""
    groups = {}
    for i, key in enumerate(map(tuple, ids[:, :depth])):
        groups.setdefault(key, []).append(i)
    chunks = []
    for members in groups.values():
        if len(members) < 2:
            continue
        sub = E[members]
        sim = sub @ sub.T
        iu = np.triu_indices(len(members), k=1)
        chunks.append(sim[iu])          # 保持 numpy,别 tolist(大组会吃几百 MB)
    if not chunks:
        return np.empty(0, dtype=np.float32), len(groups)
    return np.concatenate(chunks), len(groups)


def random_cos_pool(n_pool, E, rng):
    """抽 n_pool 个随机物品对的余弦,作为 bootstrap 的重采样池.

    只算一次,之后 bootstrap 从池里重采样 ---- 比每次重抽 2 万对快两个数量级.
    """
    n = len(E)
    i = rng.integers(0, n, size=n_pool)
    j = rng.integers(0, n, size=n_pool)
    ok = i != j                      # 去掉自己配自己
    i, j = i[ok], j[ok]
    return (E[i] * E[j]).sum(axis=1).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="item_emb.npy")
    ap.add_argument("--ids", default="semantic_ids_rqvae.json")
    ap.add_argument("--text", default="/data/zby/data/ml-1m/processed/item_text.json")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--max_pairs", type=int, default=20000,
                    help="算随机基线时最多用多少对(组太大时封顶,避免跑几十分钟)")
    ap.add_argument("--outdir", default="results/week3-4-semantic-id")
    args = ap.parse_args()

    E = normed(np.load(args.emb).astype(np.float32))
    ids = load_ids(args.ids)
    assert len(E) == len(ids), "embedding 与语义 ID 数量不一致"
    with open(args.text, encoding="utf-8") as f:
        texts = json.load(f)
    assert set(texts) == {str(i) for i in range(len(E))}, \
        "item_text.json 的键必须是连续的 0..N-1"

    rng = np.random.default_rng(42)

    # 随机基线池:一次抽好,所有深度共用(同一把尺子)
    pool = random_cos_pool(max(args.max_pairs, 50000), E, rng)
    print(f"随机配对池:{len(pool)} 对,均值 {pool.mean():.4f},"
          f"标准差 {pool.std():.4f}")

    lines = ["# 前缀语义一致性验证", "",
             f"embedding: `{args.emb}`|语义 ID: `{args.ids}`|bootstrap {args.n_boot} 次",
             "",
             "| 前缀深度 | 同前缀物品对平均余弦 | 随机配对平均余弦 | 差值 | 差值的 pair-bootstrap 95% CI | 组数 | 对数 |",
             "|---|---|---|---|---|---|---|"]

    for depth in (1, 2, 3):
        same, n_groups = group_mean_cos(ids, E, depth)
        if len(same) == 0:
            print(f"depth={depth}: 没有同前缀物品对,跳过")
            continue
        mu_same = float(same.mean())

        # ★ 点估计使用全部物品对;bootstrap 每轮最多重采样 max_pairs 对,控制时间和内存.
        n_used = min(len(same), args.max_pairs)
        mu_rand = float(pool.mean())
        boot_diff = np.empty(args.n_boot, dtype=np.float64)
        for b in range(args.n_boot):
            same_b = same[rng.integers(0, len(same), size=n_used)]
            rand_b = pool[rng.integers(0, len(pool), size=n_used)]
            boot_diff[b] = same_b.mean() - rand_b.mean()
        lo, hi = np.percentile(boot_diff, [2.5, 97.5])

        capped = "(bootstrap 已封顶)" if n_used < len(same) else ""
        lines.append(f"| 前 {depth} 位 | {mu_same:.4f} | {mu_rand:.4f} "
                     f"| **{mu_same - mu_rand:+.4f}** | [{lo:.4f}, {hi:.4f}] "
                     f"| {n_groups} | {len(same)}{capped} |")
        print(f"depth={depth}: same={mu_same:.4f} rand={mu_rand:.4f} "
              f"diff={mu_same - mu_rand:+.4f} CI=[{lo:.4f},{hi:.4f}] "
              f"groups={n_groups} pairs={len(same)} (baseline n={n_used})")

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "prefix_consistency.md"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # 候选抽样表:按组大小降序取前 20 组,每组最多 6 个物品;人工选正例并找反例.
    groups = {}
    for i, key in enumerate(map(tuple, ids[:, :2])):
        groups.setdefault(key, []).append(i)
    multi = sorted(((k, v) for k, v in groups.items() if len(v) >= 2),
                   key=lambda kv: -len(kv[1]))
    with open(os.path.join(args.outdir, "prefix_samples.csv"),
              "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prefix_c1", "prefix_c2", "item_id", "item_text", "cos_to_first"])
        for key, members in multi[:20]:
            base = E[members[0]]
            for m in members[:6]:
                writer.writerow([key[0], key[1], m, texts[str(m)],
                                 f"{float(base @ E[m]):.4f}"])
    print(f"saved {args.outdir}/prefix_consistency.md + prefix_samples.csv")
    print("→ 打开 prefix_samples.csv,人工找 2 组'同前缀但语义明显不同'的反例填进报告")


if __name__ == "__main__":
    main()