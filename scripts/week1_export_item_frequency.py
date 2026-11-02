#!/usr/bin/env python
"""从 ml-1m.inter 导出逐物品交互频次.

输出 {"0": 123, "1": 45, ...}:键是重映射后的物品 id(字符串),值是交互次数.
用途:全部清洗交互的描述性统计。评测分组另从train序列统计，避免读取未来交互。
"""
import argparse
import json
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inter", default="/data/zby/data/ml-1m/processed/ml-1m.inter")
    ap.add_argument("--out", default="item_freq.json")
    args = ap.parse_args()

    freq = Counter()
    with open(args.inter, encoding="utf-8") as f:
        header = next(f).rstrip("\r\n").split("\t")
        assert len(header) == 3, f"表头应该是 3 列,实际 {header}"
        for line in f:
            parts = line.rstrip("\r\n").split("\t")   # 服务器上是 LF,本地 Git 副本是 CRLF,两种都剥
            if len(parts) != 3:
                continue
            freq[int(parts[1])] += 1        # 第 2 列 = itemID(纯数字 token)

    n_items, total = len(freq), sum(freq.values())
    print(f"物品数 {n_items}|交互总数 {total}")
    assert n_items == 3416, f"物品数应为 3416,实际 {n_items}"
    assert total == 999611, f"交互总数应为 999611,实际 {total}"

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sorted(freq.items())}, f)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
