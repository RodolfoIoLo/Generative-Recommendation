#!/usr/bin/env python
"""把 item_text.json 编码成 item_emb.npy.

输入  item_text.json   {"0": "Toy Story (1995) | Animation|Children's|Comedy", ...}
输出  item_emb.npy     形状 (3416, D),float32,第 i 行 == 物品 id i

★ 模型走服务器本地共享目录,不联网,不下载:
    主编码器   /data/LLMs/all-MiniLM-L6-v2   → 384 维
    对照编码器 /data/LLMs/bge-m3             → 1024 维

用法:
    python scripts/week3_4_encode_items.py \
        --model /data/LLMs/all-MiniLM-L6-v2 \
        --out item_emb.npy

    python scripts/week3_4_encode_items.py \
        --model /data/LLMs/bge-m3 \
        --out item_emb_bge1024.npy --batch_size 64
"""
import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="/data/zby/data/ml-1m/processed/item_text.json")
    ap.add_argument("--model", default="/data/LLMs/all-MiniLM-L6-v2")
    ap.add_argument("--out", default="item_emb.npy")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    with open(args.text, encoding="utf-8") as f:
        texts = json.load(f)

    ids = sorted(texts.keys(), key=int)
    assert [int(k) for k in ids] == list(range(len(ids))), \
        "物品 id 不是连续的 0..N-1,先查 item_id_map.json"
    print(f"读入 {len(ids)} 条文本,示例:{texts[ids[0]]}")

    model = SentenceTransformer(args.model, device=args.device)
    emb = model.encode(
        [texts[k] for k in ids],
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,      
        show_progress_bar=True,
    ).astype(np.float32)

    assert not np.isnan(emb).any(), "编码结果里有 NaN"
    assert (np.abs(emb).sum(axis=1) > 0).all(), "有全零行"

    np.save(args.out, emb)
    print(f"saved {args.out}  shape={emb.shape}  dtype={emb.dtype}")


if __name__ == "__main__":
    main()