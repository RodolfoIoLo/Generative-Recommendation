#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务 3:数据统计。产出统计表、长尾图和抽样序列。

用法：
    python scripts/stats.py

产出:/data/zby/data/ml-1m/processed/stats/
    stats.md / stats.json / figs/*.png
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')            # 服务器没有显示器，必须放在 import pyplot 之前
import matplotlib.pyplot as plt

SEED = 42


def load_inter(path):
    """读 .inter 文件（全量交互，含 train/valid/test 三部分）。"""
    inter = pd.read_csv(path, sep='\t')
    inter.columns = ['user', 'item', 'timestamp']
    inter['user'] = inter['user'].astype(int)
    inter['item'] = inter['item'].astype(int)
    print(f'[load] {len(inter)} 条交互')
    return inter


def gini(x):
    """基尼系数：0 = 完全平均，1 = 完全集中。用来量化长尾严重程度。"""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    idx = np.arange(1, n + 1)
    return float(2 * np.sum(idx * x) / (n * np.sum(x)) - (n + 1) / n)


def compute_stats(inter):
    n_users = int(inter['user'].nunique())
    n_items = int(inter['item'].nunique())
    n_inter = int(len(inter))

    seq_len = inter.groupby('user').size()       # 每个用户的交互数
    item_freq = inter.groupby('item').size()     # 每个物品的交互数

    # 头部集中度：最热的 1% / 10% 物品，吃掉了多少比例的交互
    freq_desc = np.sort(item_freq.values)[::-1]
    cum = np.cumsum(freq_desc) / freq_desc.sum()
    top1 = max(1, int(round(0.01 * n_items)))
    top10 = max(1, int(round(0.10 * n_items)))

    s = {
        'n_users': n_users,
        'n_items': n_items,
        'n_interactions': n_inter,
        'avg_seq_len': round(float(seq_len.mean()), 2),
        'median_seq_len': int(seq_len.median()),
        'p90_seq_len': int(seq_len.quantile(0.90)),
        'max_seq_len': int(seq_len.max()),
        'min_seq_len': int(seq_len.min()),
        'sparsity': round(1 - n_inter / (n_users * n_items), 6),
        'item_freq_max': int(item_freq.max()),
        'item_freq_median': int(item_freq.median()),
        'item_freq_p90': int(item_freq.quantile(0.90)),
        'item_gini': round(gini(item_freq.values), 4),
        'items_le10_ratio': round(float((item_freq <= 10).mean()), 4),
        'items_le20_ratio': round(float((item_freq <= 20).mean()), 4),
        'items_le50_ratio': round(float((item_freq <= 50).mean()), 4),
        'top1pct_item_inter_ratio': round(float(cum[top1 - 1]), 4),
        'top10pct_item_inter_ratio': round(float(cum[top10 - 1]), 4),
        'time_min': str(pd.to_datetime(inter['timestamp'].min(), unit='s')),
        'time_max': str(pd.to_datetime(inter['timestamp'].max(), unit='s')),
    }
    return s, seq_len, item_freq


def plot_longtail(item_freq, path):
    """log-log 曲线：理想的长尾（幂律分布）在这张图上是一条直线。"""
    freq = np.sort(item_freq.values)[::-1]
    plt.figure(figsize=(6, 4))
    plt.loglog(np.arange(1, len(freq) + 1), freq, '.', markersize=3)
    plt.xlabel('Item rank (log scale)')
    plt.ylabel('Interaction count (log scale)')
    plt.title('Item popularity long-tail')
    plt.grid(True, which='both', alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'[fig] {path}')


def plot_freq_hist(item_freq, path):
    plt.figure(figsize=(6, 4))
    plt.hist(item_freq.values, bins=60)
    plt.yscale('log')
    plt.xlabel('Interactions per item')
    plt.ylabel('Number of items (log scale)')
    plt.title('Distribution of item interaction counts')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'[fig] {path}')


def plot_seq_len(seq_len, path):
    plt.figure(figsize=(6, 4))
    plt.hist(seq_len.values, bins=60)
    plt.xlabel('Sequence length (interactions per user)')
    plt.ylabel('Number of users')
    plt.title('Distribution of user sequence lengths')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'[fig] {path}')


def sample_sequences(inter, item_text, seq_len, n=3, max_show=15, seed=SEED):
    """抽 n 个活跃用户，打印带真实片名的行为序列。

    优先挑序列长的用户，因为长序列更容易看出语义连贯性。
    """
    rng = np.random.default_rng(seed)
    cand = seq_len[seq_len >= 20].index.tolist()
    if len(cand) < n:
        cand = seq_len.index.tolist()
    picked = sorted(rng.choice(cand, size=min(n, len(cand)),
                               replace=False).tolist())

    lines = []
    for u in picked:
        sub = inter[inter['user'] == u].sort_values('timestamp', kind='mergesort')
        lines.append(f'user {u}  共 {len(sub)} 次交互（只展示前 {max_show} 条）')
        for it, ts in zip(sub['item'].head(max_show), sub['timestamp'].head(max_show)):
            title = item_text.get(str(int(it)), '<unknown>')
            date = pd.to_datetime(ts, unit='s').strftime('%Y-%m-%d')
            lines.append(f'  {date}  [{int(it):>5}] {title}')
        lines.append('')
    return '\n'.join(lines)


def write_markdown(s, seq_text, path):
    """生成可读的统计表（可直接贴周报）。"""
    lines = []
    lines.append(f"# 数据统计 · {s['dataset']}\n")
    lines.append('> 由 `scripts/stats.py` 自动生成。\n')

    lines.append('## 1. 规模\n')
    lines.append('| 指标 | 值 |')
    lines.append('|---|---|')
    lines.append(f"| 用户数 | {s['n_users']:,} |")
    lines.append(f"| 物品数 | {s['n_items']:,} |")
    lines.append(f"| 交互总数 | {s['n_interactions']:,} |")
    lines.append(f"| 稀疏度 | {s['sparsity']:.4f} |\n")

    lines.append('## 2. 用户序列长度\n')
    lines.append('| 指标 | 值 |')
    lines.append('|---|---|')
    lines.append(f"| 平均 | {s['avg_seq_len']} |")
    lines.append(f"| 中位数 | {s['median_seq_len']} |")
    lines.append(f"| P90 | {s['p90_seq_len']} |")
    lines.append(f"| 最大 / 最小 | {s['max_seq_len']} / {s['min_seq_len']} |\n")

    lines.append('## 3. 物品热度分布（长尾）\n')
    lines.append('| 指标 | 值 |')
    lines.append('|---|---|')
    lines.append(f"| 最热物品交互数 | {s['item_freq_max']:,} |")
    lines.append(f"| 物品交互数中位数 | {s['item_freq_median']} |")
    lines.append(f"| 交互数 <= 10 的物品占比 | {s['items_le10_ratio']:.1%} |")
    lines.append(f"| 交互数 <= 20 的物品占比 | {s['items_le20_ratio']:.1%} |")
    lines.append(f"| 交互数 <= 50 的物品占比 | {s['items_le50_ratio']:.1%} |")
    lines.append(f"| 最热 1% 物品吃掉的交互占比 | {s['top1pct_item_inter_ratio']:.1%} |")
    lines.append(f"| 最热 10% 物品吃掉的交互占比 | {s['top10pct_item_inter_ratio']:.1%} |")
    lines.append(f"| 基尼系数 | {s['item_gini']} |\n")

    lines.append('## 4. 时间跨度\n')
    lines.append(f"{s['time_min']} ~ {s['time_max']}\n")

    lines.append('## 5. 抽样用户行为序列\n')
    for line in seq_text.split('\n'):
        lines.append('    ' + line)

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[save] {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ml-1m')
    ap.add_argument('--data-root', default='/data/zby/data')
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()

    proc = os.path.join(args.data_root, args.dataset, 'processed')
    out_dir = os.path.join(proc, 'stats')
    fig_dir = os.path.join(out_dir, 'figs')
    os.makedirs(fig_dir, exist_ok=True)

    inter = load_inter(os.path.join(proc, f'{args.dataset}.inter'))
    with open(os.path.join(proc, 'item_text.json'), encoding='utf-8') as f:
        item_text = json.load(f)

    s, seq_len, item_freq = compute_stats(inter)
    s['dataset'] = args.dataset

    plot_longtail(item_freq, os.path.join(fig_dir, 'item_freq_loglog.png'))
    plot_freq_hist(item_freq, os.path.join(fig_dir, 'item_freq_hist.png'))
    plot_seq_len(seq_len, os.path.join(fig_dir, 'seq_len_hist.png'))

    seq_text = sample_sequences(inter, item_text, seq_len, seed=args.seed)

    with open(os.path.join(out_dir, 'stats.json'), 'w', encoding='utf-8') as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    print(f'[save] {os.path.join(out_dir, "stats.json")}')

    write_markdown(s, seq_text, os.path.join(out_dir, 'stats.md'))

    print('\n' + '=' * 64)
    for k, v in s.items():
        print(f'{k:34s} {v}')
    print('=' * 64)
    print('\n抽样序列：\n' + seq_text)


if __name__ == '__main__':
    main()