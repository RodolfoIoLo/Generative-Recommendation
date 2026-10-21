#!/usr/bin/env python
"""导出阶段二 → 阶段三的接口文件,并自检.

用法:
    python scripts/export_semantic.py --ids semantic_ids_rqvae.json \
      --emb item_emb.npy --quantizer_checkpoint saved/rqvae_ml1m.pt \
      --processed /data/zby/data/ml-1m/processed --outdir exported

产出(outdir 下):
    semantic_ids.json                 item_id -> [c1,c2,c3,c4]
    train/valid/test.semantic.jsonl   与第一周同结构,物品 id 换成 4 元组
    token_vocab.json                  语义 token -> 词表下标
    meta.json                         协议,归一化,产物哈希,L / K / 词表大小 / 来源
"""
import argparse
import hashlib
import json
import os

import numpy as np


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path, chunk_size=1024 * 1024):
    """流式计算文件哈希,避免把较大的 checkpoint 一次性读进内存."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def build_vocab(n_levels, codebook_size, specials=("<pad>", "<bos>", "<eos>")):
    """每层独立占 codebook_size 个 token:第 l 层码字 k -> 下标 len(specials)+l*K+k."""
    token2id = {s: i for i, s in enumerate(specials)}
    n = len(specials)
    for l in range(n_levels):
        for k in range(codebook_size):
            token2id[f"{l}_{k}"] = n + l * codebook_size + k
    return token2id, n + n_levels * codebook_size


def convert(items, sid):
    return [sid[str(i)] for i in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="semantic_ids_rqvae.json")
    ap.add_argument("--processed", default="/data/zby/data/ml-1m/processed")
    ap.add_argument("--outdir", default="exported")
    ap.add_argument("--codebook_size", type=int, default=256)
    ap.add_argument("--rq_levels", type=int, default=3,
                    help="RQ 量化层数 L(不含消歧位)")
    ap.add_argument("--dedup", dest="dedup", action="store_true", default=True,
                    help="语义 ID 末位是消歧码(TIGER 口径,默认开)")
    ap.add_argument("--no_dedup", dest="dedup", action="store_false")
    ap.add_argument("--encoder", default="/data/LLMs/all-MiniLM-L6-v2")
    ap.add_argument("--emb", default="item_emb.npy")
    ap.add_argument("--embedding_dim", type=int, default=384)
    ap.add_argument("--embedding_normalization", choices=("l2", "none"),
                    default="l2")
    ap.add_argument("--quantizer_checkpoint", default="saved/rqvae_ml1m.pt")
    args = ap.parse_args()

    # 冻结的不只是语义 ID,还要冻结生成它们的输入向量与量化器.
    # 这里验证 v3.1 协议中的 MiniLM 输出形状和 L2 归一化约定.
    emb = np.load(args.emb, mmap_mode="r")
    assert emb.shape == (3416, args.embedding_dim), (
        f"嵌入形状应为 (3416, {args.embedding_dim}),实际 {emb.shape}")
    assert np.isfinite(emb).all(), "嵌入含 NaN/Inf"
    if args.embedding_normalization == "l2":
        norms = np.linalg.norm(emb, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), (
            f"声明为 L2 归一化,但范数范围为 [{norms.min():.8f}, {norms.max():.8f}]")
    assert os.path.isfile(args.quantizer_checkpoint), (
        f"量化器 checkpoint 不存在:{args.quantizer_checkpoint}")
    emb_sha256 = sha256_file(args.emb)
    quantizer_sha256 = sha256_file(args.quantizer_checkpoint)

    sid = load_json(args.ids)
    n_items = len(sid)
    assert n_items == 3416, f"语义 ID 数应为 3416,实际 {n_items}"
    assert set(sid) == {str(i) for i in range(n_items)}, \
        "语义 ID 的键必须是连续的 0..3415"

    # ---- 关键:区分"RQ 层数 L"和"语义 ID 长度" ----
    # L = 3(三层残差量化);语义 ID 长度 = L + 1 = 4(末位是消歧码).
    # 词表必须按"语义 ID 长度"算:256 × 4 = 1024 个物品 token,
    # 因为第 4 位消歧码也占一个完整的 256 宽码本位(TIGER §4 原文).
    sid_len = len(next(iter(sid.values())))
    expect_len = args.rq_levels + (1 if args.dedup else 0)
    assert all(len(v) == sid_len for v in sid.values()), "语义 ID 长度不一致"
    assert sid_len == expect_len, (
        f"语义 ID 长度 {sid_len} 与声明不符(rq_levels={args.rq_levels}, "
        f"dedup={args.dedup} -> 期望 {expect_len})."
        f"检查 --ids 指向的文件和 --rq_levels/--no_dedup 是否一致.")
    assert all(type(code) is int for v in sid.values() for code in v), \
        "语义 ID 中存在非整数码"
    code_array = np.asarray([sid[str(i)] for i in range(n_items)], dtype=np.int64)
    assert ((0 <= code_array) & (code_array < args.codebook_size)).all(), \
        f"语义码必须落在 [0, {args.codebook_size - 1}]"
    uniq = {tuple(v) for v in code_array.tolist()}
    assert len(uniq) == n_items, f"语义 ID 有重复:{n_items - len(uniq)} 个"

    # 消歧码是组内序号,最大可能等于"最大冲突组大小 - 1",
    # 必须能塞进第 4 位那 256 个码字里.
    if args.dedup:
        max_dedup = max(int(v[-1]) for v in sid.values())
        assert max_dedup < args.codebook_size, (
            f"最大消歧码 {max_dedup} >= 码本容量 {args.codebook_size},"
            f"说明有 {max_dedup + 1} 个物品撞到了完全相同的 {args.rq_levels} 位前缀."
            f"先去看 codebook_report.md 的冲突率,换编码器或加层.")

    token2id, vocab_size = build_vocab(sid_len, args.codebook_size)
    assert len(token2id) == vocab_size, "词表下标有重复"

    os.makedirs(args.outdir, exist_ok=True)
    source_ids_sha256 = sha256_file(args.ids)
    source_split_sha256 = {}

    with open(os.path.join(args.outdir, "semantic_ids.json"),
              "w", encoding="utf-8") as f:
        json.dump(sid, f)

    for split in ("train", "valid", "test"):
        src = os.path.join(args.processed, f"{split}.jsonl")
        dst = os.path.join(args.outdir, f"{split}.semantic.jsonl")
        source_split_sha256[split] = sha256_file(src)
        n_in = n_out = 0
        with open(src, encoding="utf-8") as fi, \
                open(dst, "w", encoding="utf-8") as fo:
            for line in fi:
                row = json.loads(line)
                n_in += 1
                if "items" in row:
                    row["items"] = convert(row["items"], sid)
                if "history" in row:
                    row["history"] = convert(row["history"], sid)
                if "target" in row:
                    row["target"] = sid[str(row["target"])]
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_out += 1
        assert n_in == n_out == 6040, f"{split} 行数应为 6040,实际 {n_in}->{n_out}"
        print(f"{split}.semantic.jsonl  {n_out} 行")

    n_item_tokens = sid_len * args.codebook_size
    with open(os.path.join(args.outdir, "token_vocab.json"),
              "w", encoding="utf-8") as f:
        json.dump({
            "pad_token": "<pad>", "bos_token": "<bos>", "eos_token": "<eos>",
            "rq_levels": args.rq_levels,              # 3
            "semantic_id_length": sid_len,            # 4
            "has_dedup_token": args.dedup,            # True
            "n_item_tokens": n_item_tokens,           # 1024
            "codebook_size": args.codebook_size,      # 256
            "vocab_size": vocab_size,                 # 1027
            "token2id": token2id,
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "protocol_version": "v3.1",
            "dataset": "ml-1m", "n_items": n_items,
            "encoder": args.encoder,
            "embedding_file": args.emb,
            "embedding_dim": args.embedding_dim,
            "embedding_normalization": args.embedding_normalization,
            "embedding_sha256": emb_sha256,
            "quantizer_checkpoint": args.quantizer_checkpoint,
            "quantizer_sha256": quantizer_sha256,
            "source_ids_sha256": source_ids_sha256,
            "source_split_sha256": source_split_sha256,
            "rq_levels": args.rq_levels,
            "semantic_id_length": sid_len,
            "has_dedup_token": args.dedup,
            "codebook_size": args.codebook_size,
            "n_item_tokens": n_item_tokens,
            "source_ids_file": args.ids,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nRQ 层数 L = {args.rq_levels}|语义 ID 长度 = {sid_len}"
          f"(含消歧位:{args.dedup})")
    print(f"词表:{sid_len} × {args.codebook_size} = {n_item_tokens} 个物品 token"
          f"(+3 特殊 token,共 {vocab_size})")
    print(f"嵌入 SHA-256:{emb_sha256}")
    print(f"量化器 SHA-256:{quantizer_sha256}")
    print(f"全部产出已写入 {args.outdir}/")


if __name__ == "__main__":
    main()