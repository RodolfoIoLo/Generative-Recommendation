#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ml-1m 预处理：原始评分文件 -> train/valid/test + 映射表 + 物品文本

用法：
    python scripts/preprocess.py --stage load   # 只验证读取，不写文件
    python scripts/preprocess.py                # 跑完整流程

产出目录：/data/zby/data/ml-1m/processed/
"""

import argparse
import csv
import json
import os
import random

import numpy as np
import pandas as pd

SEED = 42


def set_seed(seed):
    """固定所有随机性，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)


# ==========================================================================
# 1. 读原始数据
# ==========================================================================
def load_ml1m(raw_dir):
    """读 ml-1m 的 ratings.dat 和 movies.dat。

    两个坑：
      1. 分隔符是 '::'（两个字符），多字符分隔符必须 engine='python'
      2. movies.dat 是 latin-1 编码，用 utf-8 读会抛 UnicodeDecodeError
    """
    ratings = pd.read_csv(
        os.path.join(raw_dir, 'ratings.dat'),
        sep='::', engine='python', header=None, quoting=csv.QUOTE_NONE,
        names=['user', 'item', 'rating', 'timestamp'],
        encoding='latin-1',
        dtype={'user': int, 'item': int, 'rating': int, 'timestamp': int},
    )
    movies = pd.read_csv(
        os.path.join(raw_dir, 'movies.dat'),
        sep='::', engine='python', header=None, quoting=csv.QUOTE_NONE,
        names=['item', 'title', 'genres'],
        encoding='latin-1',
    )
    print(f'[load] ratings.shape = {ratings.shape}')
    print(f'[load] movies.shape  = {movies.shape}')
    print(f'[load] ratings 前 3 行:\n{ratings.head(3)}')
    print(f'[load] movies  前 3 行:\n{movies.head(3)}')
    t_min = pd.to_datetime(ratings['timestamp'].min(), unit='s')
    t_max = pd.to_datetime(ratings['timestamp'].max(), unit='s')
    print(f'[load] 时间范围 {t_min} ~ {t_max}')
    return ratings, movies


# ==========================================================================
# 2. 转隐式反馈（丢掉评分列）
# ==========================================================================
def to_implicit(ratings):
    """只保留 (user, item, timestamp)。

    决策：ml-1m 的 1~5 分全部算「有交互」，不设阈值。
    """
    n0 = len(ratings)
    df = ratings[['user', 'item', 'timestamp']].drop_duplicates()
    print(f'[implicit] {n0} → {len(df)} 条（丢评分列 + 去重）')
    return df


# ==========================================================================
# 3. k-core 迭代过滤  ★ 重点
# ==========================================================================
def kcore_filter(df, k):
    """迭代过滤，直到所有用户和物品的交互数都 >= k。

    为什么必须迭代：删掉冷门物品后，有些用户的交互数会掉到 k 以下；
    再删这些用户，又会有物品掉下去。单次过滤的结果是错的。
    """
    n_round = 0
    while True:
        n_round += 1
        n_before = len(df)
        user_cnt = df['user'].value_counts()
        item_cnt = df['item'].value_counts()
        df = df[df['user'].isin(user_cnt[user_cnt >= k].index) &
                df['item'].isin(item_cnt[item_cnt >= k].index)]
        n_after = len(df)
        print(f'[kcore] 第 {n_round} 轮: {n_before} → {n_after}')
        if n_after == n_before:          # 没有变化 → 收敛
            break
    print(f'[kcore] 收敛，共 {n_round} 轮')
    return df.reset_index(drop=True)


# ==========================================================================
# 4. ID 重映射
# ==========================================================================
def remap_ids(df):
    """原始 ID → 0..N-1。

    用 sorted() 而不是 set：set 的迭代顺序每次可能不同，映射会不可复现。
    """
    users = sorted(df['user'].unique().tolist())
    items = sorted(df['item'].unique().tolist())
    user_map = {int(u): i for i, u in enumerate(users)}
    item_map = {int(x): j for j, x in enumerate(items)}

    df = df.copy()
    df['user'] = df['user'].map(user_map).astype(int)
    df['item'] = df['item'].map(item_map).astype(int)
    print(f'[remap] {len(user_map)} 用户, {len(item_map)} 物品')

    assert df['user'].max() == len(user_map) - 1, 'user ID 不连续'
    assert df['item'].max() == len(item_map) - 1, 'item ID 不连续'
    return df, user_map, item_map


# ==========================================================================
# 5. 稳定排序 + leave-one-out 划分
# ==========================================================================
def split_loo(df):
    """每个用户按时间排序：最后 1 个 → test，倒数第 2 个 → valid，其余 → train。

    排序必须稳定（kind='mergesort'），并用 item 做二级排序键，
    否则时间戳相同的记录顺序会变，结果不可复现。
    """
    df = df.sort_values(['user', 'timestamp', 'item'],
                        kind='mergesort').reset_index(drop=True)
    df['rank_from_end'] = df.groupby('user').cumcount(ascending=False)

    train = df[df['rank_from_end'] >= 2].copy()
    valid = df[df['rank_from_end'] == 1].copy()
    test = df[df['rank_from_end'] == 0].copy()

    print(f'[split] train {len(train)} / valid {len(valid)} / test {len(test)}')

    assert len(train) + len(valid) + len(test) == len(df), '划分后总数不守恒'
    assert valid['user'].is_unique, 'valid 里有重复用户'
    assert test['user'].is_unique, 'test 里有重复用户'
    assert train.groupby('user').size().min() >= 1, '有用户的 train 为空'
    return train, valid, test


# ==========================================================================
# 6. 拼成 (history, target) 并落盘
# ==========================================================================
def build_sequences(train, valid, test):
    """拼成模型要的「接龙题目」。"""
    train_hist = (train.sort_values(['user', 'timestamp'], kind='mergesort')
                       .groupby('user')['item'].agg(list).to_dict())
    valid_target = valid.set_index('user')['item'].to_dict()
    test_target = test.set_index('user')['item'].to_dict()

    train_records = [{'user': int(u), 'items': [int(x) for x in train_hist[u]]}
                     for u in sorted(train_hist)]

    valid_records = [{'user': int(u),
                      'history': [int(x) for x in train_hist[u]],
                      'target': int(valid_target[u])}
                     for u in sorted(valid_target)]

    # test 的 history = train + valid，且不含 test 自己（防数据泄漏）
    test_records = [{'user': int(u),
                     'history': [int(x) for x in train_hist[u]] + [int(valid_target[u])],
                     'target': int(test_target[u])}
                    for u in sorted(test_target)]

    return train_records, valid_records, test_records


def write_jsonl(records, path):
    """JSONL：每行一个 JSON。"""
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[save] {path}  ({len(records)} 行)')


def write_json(obj, path):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f'[save] {path}')


def write_inter(df, path):
    """RecBole 的 .inter 格式，后面跑 SASRec 基线直接读这个。

    ★ 排序键必须和 split_loo 完全一致（user, timestamp, item）。
      如果这里只用 (user, timestamp)，同一时间戳内的顺序会保留原始文件顺序，
      而 split_loo 用的是 item 作二级键 —— 两边算出来的 leave-one-out 切点会不同，
      结果是 RecBole 跑出来的基线和 jsonl 训练出来的模型评测在不同的测试集上。
      实测：不一致时 6040 个用户里有 1561 个（25.8%）的 test 目标物品会变。
    """
    out = df.sort_values(['user', 'timestamp', 'item'], kind='mergesort')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('userID:token\titemID:token\ttimestamp:float\n')
        for u, i, t in zip(out['user'], out['item'], out['timestamp']):
            f.write(f'{int(u)}\t{int(i)}\t{float(t)}\n')
    print(f'[save] {path}  ({len(out)} 行)')


def build_item_text(movies, item_map, path):
    """item id → 文本描述，供第 3 周构建语义标识符用。"""
    text = {}
    skipped = 0
    for raw_id, title, genres in zip(movies['item'], movies['title'], movies['genres']):
        raw_id = int(raw_id)
        if raw_id not in item_map:        # 被 5-core 过滤掉的物品
            skipped += 1
            continue
        text[str(item_map[raw_id])] = f'{title} | {genres}'
    write_json(text, path)
    print(f'[item_text] 写出 {len(text)} 条，跳过被过滤的 {skipped} 条')
    return text


# ==========================================================================
# 主流程
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ml-1m')
    ap.add_argument('--data-root', default='/data/zby/data')
    ap.add_argument('--k', type=int, default=5, help='k-core 的 k')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--stage', choices=['load', 'all'], default='all',
                    help='load = 只验证能不能读进来')
    args = ap.parse_args()

    set_seed(args.seed)

    raw_dir = os.path.join(args.data_root, args.dataset, 'raw')
    out_dir = os.path.join(args.data_root, args.dataset, 'processed')
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 64)
    print(f'数据集   : {args.dataset}')
    print(f'原始目录 : {raw_dir}')
    print(f'输出目录 : {out_dir}')
    print(f'k-core k : {args.k}')
    print(f'随机种子 : {args.seed}')
    print('=' * 64)

    ratings, movies = load_ml1m(raw_dir)
    if args.stage == 'load':
        print('\n[stage=load] 只验证读取，到此结束。')
        return

    n_raw_users = int(ratings['user'].nunique())
    n_raw_items = int(ratings['item'].nunique())

    df = to_implicit(ratings)
    df = kcore_filter(df, args.k)
    df, user_map, item_map = remap_ids(df)
    train, valid, test = split_loo(df)

    train_rec, valid_rec, test_rec = build_sequences(train, valid, test)

    write_jsonl(train_rec, os.path.join(out_dir, 'train.jsonl'))
    write_jsonl(valid_rec, os.path.join(out_dir, 'valid.jsonl'))
    write_jsonl(test_rec, os.path.join(out_dir, 'test.jsonl'))
    write_json(user_map, os.path.join(out_dir, 'user_id_map.json'))
    write_json(item_map, os.path.join(out_dir, 'item_id_map.json'))
    write_inter(df, os.path.join(out_dir, f'{args.dataset}.inter'))
    build_item_text(movies, item_map, os.path.join(out_dir, 'item_text.json'))

    log = {
        'dataset': args.dataset,
        'k_core': args.k,
        'seed': args.seed,
        'raw': {'users': n_raw_users, 'items': n_raw_items,
                'interactions': int(len(ratings))},
        'after_kcore': {'users': int(df['user'].nunique()),
                        'items': int(df['item'].nunique()),
                        'interactions': int(len(df))},
        'split': {'train': int(len(train)), 'valid': int(len(valid)),
                  'test': int(len(test))},
    }
    write_json(log, os.path.join(out_dir, 'preprocess_log.json'))
    print('\n处理完成。')


if __name__ == '__main__':
    main()
