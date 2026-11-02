#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
校验 leave-one-out 划分:.inter 与 train/valid/test.jsonl 是否严格一致.

只用标准库,不需要 pandas,本地也能跑(方便在 Windows 上先自查).

用法:
    python scripts/week1_verify_split.py --data-dir .check/ml-1m/processed

检查 5 件事:
  1) .inter 里每个用户的交互是否按时间单调不减(用户行为按时间排序)
  2) (user, item) 是否唯一(ml-1m 不应有重复交互)
  3) test.target  == 该用户最后一个交互
  4) valid.target == 该用户倒数第二个交互
  5) test.history / valid.history 是否等于"去掉尾部 N 个"的前缀(无数据泄漏)
"""

import argparse
import json
import os
import sys


def load_inter(path):
    """读 .inter -> {user: [(item, timestamp), ...]},保持文件顺序."""
    seq = {}
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline()
        fields = header.rstrip('\n').split('\t')
        assert fields[0].startswith('userID') and fields[1].startswith('itemID'), \
            f'.inter 表头不对: {header!r}'
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            u, i, t = int(parts[0]), int(parts[1]), float(parts[2])
            seq.setdefault(u, []).append((i, t))
    return seq


def load_jsonl(path, key):
    """读 jsonl -> {user: 该行 dict}."""
    out = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            out[obj['user']] = obj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True, help='processed 目录')
    ap.add_argument('--dataset', default='ml-1m')
    args = ap.parse_args()

    d = args.data_dir
    inter_path = os.path.join(d, f'{args.dataset}.inter')
    for name in (inter_path, os.path.join(d, 'train.jsonl'),
                 os.path.join(d, 'valid.jsonl'), os.path.join(d, 'test.jsonl')):
        if not os.path.exists(name):
            print(f'[FAIL] 找不到文件: {name}')
            sys.exit(1)

    seq = load_inter(inter_path)
    train = load_jsonl(os.path.join(d, 'train.jsonl'), 'items')
    valid = load_jsonl(os.path.join(d, 'valid.jsonl'), 'history')
    test = load_jsonl(os.path.join(d, 'test.jsonl'), 'history')

    errors = []
    n_inter = sum(len(v) for v in seq.values())

    # --- 1) 时间单调 + 2) (user,item) 唯一 ---
    bad_time = []
    dup_pairs = 0
    for u, rows in seq.items():
        for k in range(1, len(rows)):
            if rows[k][1] < rows[k - 1][1]:
                bad_time.append(u)
                break
        if len({i for i, _ in rows}) != len(rows):
            dup_pairs += 1
    if bad_time:
        errors.append(f'{len(bad_time)} 个用户的时间戳不是单调不减,例如 user={bad_time[0]}')

    # --- 3)~5) 逐用户对齐 ---
    users = sorted(seq.keys())
    if sorted(train.keys()) != users:
        errors.append('train.jsonl 的用户集合与 .inter 不一致')
    if sorted(valid.keys()) != users:
        errors.append('valid.jsonl 的用户集合与 .inter 不一致')
    if sorted(test.keys()) != users:
        errors.append('test.jsonl 的用户集合与 .inter 不一致')

    bad_last, bad_second, bad_test_hist, bad_valid_hist, bad_train = [], [], [], [], []
    for u in users:
        items = [i for i, _ in seq[u]]
        if u not in train or u not in valid or u not in test:
            continue
        if items[-1] != test[u]['target']:
            bad_last.append(u)
        if items[-2] != valid[u]['target']:
            bad_second.append(u)
        if items[:-1] != test[u]['history']:
            bad_test_hist.append(u)
        if items[:-2] != valid[u]['history']:
            bad_valid_hist.append(u)
        if items[:-2] != train[u]['items']:
            bad_train.append(u)

    def rep(name, bad, msg):
        if bad:
            errors.append(f'{name}: {len(bad)}/{len(users)} 个用户不符(例如 user={bad[0]})---- {msg}')

    rep('test.target', bad_last, '最后一个交互没被当作测试目标')
    rep('valid.target', bad_second, '倒数第二个交互没被当作验证目标')
    rep('test.history', bad_test_hist, 'test 的 history 不等于"去掉最后 1 个"的前缀')
    rep('valid.history', bad_valid_hist, 'valid 的 history 不等于"去掉最后 2 个"的前缀')
    rep('train.items', bad_train, 'train 序列不等于"去掉最后 2 个"的前缀')

    # --- 汇总 ---
    print('=' * 60)
    print(f'目录       : {d}')
    print(f'用户数     : {len(users)}')
    print(f'交互总数   : {n_inter}')
    print(f'train/valid/test 行数: {len(train)} / {len(valid)} / {len(test)}')
    print(f'train+valid+test = {len(train) + len(valid) + len(test)}(应等于 用户数×3)')
    print(f'时间单调   : {"OK" if not bad_time else "FAIL"}')
    print(f'(user,item) 唯一: {"OK" if dup_pairs == 0 else f"FAIL({dup_pairs} 个用户有重复)"}')
    print('=' * 60)
    if errors:
        print('[FAIL] 发现以下问题:')
        for e in errors:
            print('  -', e)
        sys.exit(1)
    print('[PASS] 划分与 .inter 严格一致,无数据泄漏.')


if __name__ == '__main__':
    main()
