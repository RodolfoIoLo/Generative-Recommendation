#!/usr/bin/env python
"""构建语义标识符：RQ-KMeans（baseline）/ RQ-VAE（主方案）+ 冲突消解。

用法:
    python scripts/week3_4_build_semantic_ids.py --emb item_emb.npy --method kmeans
    python scripts/week3_4_build_semantic_ids.py --emb item_emb.npy --method rqvae --epochs 2000

输出  semantic_ids.json   {"0": [37,128,5,0], ...}  长度 4,第 4 位是消歧码(无冲突补 0)
"""
from week1_paths import result_path, output_path
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.cluster import KMeans


# ------------------------------------------------------------------ RQ-Kmeans
def rq_kmeans(X, n_levels, codebook_size, seed=42):
    """每层对当前残差做 k-means.K < n,sklearn 会重分配空簇 → 利用率恒 100%."""
    codes = np.zeros((len(X), n_levels), dtype=np.int64)
    codebooks = []
    R = X.copy()
    for l in range(n_levels):
        km = KMeans(n_clusters=codebook_size, n_init=10,
                    random_state=seed + l).fit(R)
        codes[:, l] = km.labels_
        centers = km.cluster_centers_.astype(np.float32)
        codebooks.append(centers)
        R = R - centers[km.labels_]
        print(f"  level {l}: 用到 {len(np.unique(km.labels_))}/{codebook_size} 个码字")
    return codes, np.stack(codebooks, axis=0)


# -------------------------------------------------------------------- RQ-VAE
class RQVAE(nn.Module):
    """结构照 TIGER §4:编码器 D→512→256→128→latent,L 层码本 K×latent,解码器反向.
    损失照 TIGER §3.1:L = L_recon + Σ_l ( ||sg[r_l]-e||² + β||r_l-sg[e]||² )
    """

    def __init__(self, in_dim, n_levels=3, codebook_size=256, latent=32, beta=0.25):
        super().__init__()
        self.n_levels = n_levels
        self.codebook_size = codebook_size
        self.latent = latent
        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, in_dim),
        )
        self.codebooks = nn.ModuleList(
            [nn.Embedding(codebook_size, latent) for _ in range(n_levels)]
        )
        # 占位值；正式训练前由编码器潜空间的逐层 K-Means 质心覆盖。
        for cb in self.codebooks:
            nn.init.zeros_(cb.weight)

    def quantize(self, z):
        residual = z
        quantized_sum = torch.zeros_like(z)
        codebook_loss = z.new_zeros(())
        commitment_loss = z.new_zeros(())
        codes = []
        for l in range(self.n_levels):
            e = self.codebooks[l].weight                     # [K, latent]
            idx = torch.cdist(residual, e).argmin(dim=1)     # [B]
            ec = e[idx]                                      # [B, latent]
            codes.append(idx)
            quantized_sum = quantized_sum + ec
            # 每个向量内部求平方和、再对 batch 求均值。这样 384 维重建项与
            # 32 维量化项都对应平方 L2 距离，不会因维度不同而隐式改权重。
            codebook_loss = codebook_loss \
                + ((residual.detach() - ec) ** 2).sum(dim=1).mean()
            commitment_loss = commitment_loss \
                + self.beta * ((residual - ec.detach()) ** 2).sum(dim=1).mean()
            residual = residual - ec.detach()                # ★ 必须 detach
        return (quantized_sum, torch.stack(codes, dim=1),
                codebook_loss, commitment_loss)

    def forward(self, x):
        z = self.encoder(x)
        quantized_sum, codes, codebook_loss, commitment_loss = self.quantize(z)
        z_q = z + (quantized_sum - z).detach()               # ★ 直通估计器
        x_hat = self.decoder(z_q)
        recon_loss = ((x_hat - x) ** 2).sum(dim=1).mean()
        return x_hat, codes, recon_loss, codebook_loss, commitment_loss


def code_stats(codes, codebook_size):
    """返回逐层使用量、困惑度、完整前缀唯一数。"""
    codes = np.asarray(codes)
    util, ppl = [], []
    for l in range(codes.shape[1]):
        counts = np.bincount(codes[:, l], minlength=codebook_size)
        p = counts[counts > 0] / len(codes)
        util.append(int((counts > 0).sum()))
        ppl.append(float(np.exp(-(p * np.log(p)).sum())))
    unique = len({tuple(row) for row in codes.tolist()})
    return util, ppl, unique


def evaluate_rqvae(model, Xt, total_var):
    model.eval()
    with torch.no_grad():
        x_hat, codes, _, codebook, commitment = model(Xt)
        sse = float(((Xt - x_hat) ** 2).sum())
    return {
        "r2": 1.0 - sse / total_var,
        "mean_mse": sse / Xt.numel(),
        "codebook_loss": float(codebook),
        "commitment_loss": float(commitment),
        "codes": codes.cpu().numpy(),
    }


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_autoencoder(model, Xt, total_var, epochs, batch_size, lr,
                      log_every, dev):
    """先学习稳定的连续潜空间，避免随机码本把编码器拉到常数解。"""
    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    n = len(Xt)
    best_r2, best_state = -float("inf"), None
    print(f"[AE 预训练] epochs={epochs} lr={lr:g}")
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch_size):
            xb = Xt[perm[i:i + batch_size]]
            x_hat = model.decoder(model.encoder(xb))
            loss = ((x_hat - xb) ** 2).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        if ep % log_every == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                x_hat = model.decoder(model.encoder(Xt))
                sse = float(((Xt - x_hat) ** 2).sum())
            r2 = 1.0 - sse / total_var
            mean_mse = sse / Xt.numel()
            print(f"AE epoch {ep:5d} | mean-MSE {mean_mse:.8f} | R2 {r2:.4f}")
            if r2 > best_r2:
                best_r2, best_state = r2, clone_state(model)

    if best_state is None:
        raise RuntimeError("AE 预训练没有产生可用 checkpoint")
    model.load_state_dict(best_state)
    if best_r2 <= 0:
        raise RuntimeError(f"AE 预训练 R2={best_r2:.4f}，没有优于均值预测器")
    print(f"[AE 预训练完成] best R2={best_r2:.4f}")
    return best_r2


def init_codebooks_with_kmeans(model, Xt, seed, n_init):
    """在全部潜向量上逐层做残差 K-Means，并写入各层码本。"""
    model.eval()
    with torch.no_grad():
        z = model.encoder(Xt).cpu().numpy().astype(np.float32)
    codes = np.zeros((len(z), model.n_levels), dtype=np.int64)
    centers = []
    residual = z.copy()
    for l in range(model.n_levels):
        km = KMeans(n_clusters=model.codebook_size, n_init=n_init,
                    random_state=seed + l).fit(residual)
        codes[:, l] = km.labels_
        center = km.cluster_centers_.astype(np.float32)
        centers.append(center)
        residual = residual - center[km.labels_]
    with torch.no_grad():
        for l, cb in enumerate(model.codebooks):
            cb.weight.copy_(torch.from_numpy(centers[l]).to(cb.weight.device))
    util, ppl, unique = code_stats(codes, model.codebook_size)
    print(f"[潜空间 K-Means 初始化] 利用率 {util} | "
          f"困惑度 {[round(v, 1) for v in ppl]} | 唯一 ID {unique}")


def train_rqvae(X, n_levels, codebook_size, latent=32, beta=0.25,
                epochs=2000, batch_size=256, lr=1e-4, seed=42,
                log_every=100, device="cpu", pretrain_epochs=200,
                pretrain_lr=1e-3, init_n_init=10,
                min_utilization=0.9):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"请求了 {device},但 torch.cuda.is_available() 为 False")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dev = torch.device(device)

    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    model = RQVAE(X.shape[1], n_levels, codebook_size, latent, beta).to(dev)
    n = len(Xt)
    total_var = float(((X - X.mean(axis=0)) ** 2).sum())

    train_autoencoder(model, Xt, total_var, pretrain_epochs, batch_size,
                      pretrain_lr, min(log_every, 50), dev)
    init_codebooks_with_kmeans(model, Xt, seed, init_n_init)

    # 初始化完成后的模型也是候选 checkpoint；联合训练若退化，不会覆盖它。
    best_any = None
    best_qualified = None

    def consider(epoch):
        nonlocal best_any, best_qualified
        metrics = evaluate_rqvae(model, Xt, total_var)
        util, ppl, unique = code_stats(metrics["codes"], codebook_size)
        metrics.update(epoch=epoch, util=util, ppl=ppl, unique=unique,
                       state=clone_state(model))
        print(f"RQ epoch {epoch:5d} | mean-MSE {metrics['mean_mse']:.8f} "
              f"| R2 {metrics['r2']:.4f} "
              f"| codebook {metrics['codebook_loss']:.3e} "
              f"| beta*commit {metrics['commitment_loss']:.3e} "
              f"| 利用率 {util} | 困惑度 {[round(v, 1) for v in ppl]} "
              f"| 唯一 ID {unique}")
        if best_any is None or metrics["r2"] > best_any["r2"]:
            best_any = metrics
        threshold = codebook_size * min_utilization
        if min(util) >= threshold:
            if best_qualified is None or metrics["r2"] > best_qualified["r2"]:
                best_qualified = metrics

    consider(0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"[RQ-VAE 联合训练] epochs={epochs} lr={lr:g} "
          f"合格利用率阈值={min_utilization:.0%}")
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            _, _, recon, codebook, commitment = model(Xt[idx])
            opt.zero_grad()
            (recon + codebook + commitment).backward()
            opt.step()

        if ep % log_every == 0 or ep == epochs:
            consider(ep)

    best = best_qualified if best_qualified is not None else best_any
    model.load_state_dict(best["state"])
    if best_qualified is None:
        print("WARNING: 没有 checkpoint 同时满足所有层的利用率阈值，回退到最高 R2")
    print(f"[选择 checkpoint] epoch={best['epoch']} | R2={best['r2']:.4f} "
          f"| 利用率={best['util']} | 唯一 ID={best['unique']}")
    return model, best["codes"], best


# ---------------------------------------------------------------- 冲突消解
def dedup(codes):
    """TIGER §3.1 Handling Collisions:前 L 位相同的物品追加第 L+1 位组内序号;
    无冲突补 0.返回 [N, L+1]."""
    codes = np.asarray(codes)
    n, L = codes.shape
    seen, out = {}, np.zeros((n, L + 1), dtype=np.int64)
    out[:, :L] = codes
    for i in range(n):
        key = tuple(codes[i].tolist())
        k = seen.get(key, 0)
        out[i, L] = k
        seen[key] = k + 1
    return out


def report(codes, tag, codebook_size, has_dedup=False):
    """逐列统计.has_dedup=True 时最后一列是消歧码,单独标注.

    ★ 注意:消歧码那一列的"利用率"没有码本含义 ---- 它只取 0..最大组大小-1,
      利用率低是正常的,不要当成码本塌缩.真正的码本质量只看前 n_levels 列.
    """
    codes = np.asarray(codes)
    n = len(codes)
    uniq = len({tuple(r) for r in codes.tolist()})
    print(f"[{tag}] 物品 {n}|唯一 ID {uniq}|冲突率 {1 - uniq / n:.4f}")
    for l in range(codes.shape[1]):
        col = codes[:, l]
        u = len(np.unique(col))
        p = np.bincount(col) / n
        p = p[p > 0]
        ppl = float(np.exp(-(p * np.log(p)).sum()))
        mark = "  ← 消歧码,不是码本" if (has_dedup and l == codes.shape[1] - 1) else ""
        print(f"  level {l}: 利用率 {u}/{codebook_size}|困惑度 {ppl:.1f}{mark}")


def parse_args():
    # 先读 config 路径,再把 YAML 值设为 argparse 默认值;显式 CLI 参数优先级最高.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config/rqvae_ml1m.yaml")
    known, _ = pre.parse_known_args()
    with open(known.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--emb", default="item_emb.npy")
    ap.add_argument("--method", choices=["kmeans", "rqvae"], default="kmeans")
    ap.add_argument("--in_dim", type=int, default=cfg.get("in_dim", 384))
    ap.add_argument("--n_levels", type=int, default=cfg.get("n_levels", 3))
    ap.add_argument("--codebook_size", type=int, default=cfg.get("codebook_size", 256))
    ap.add_argument("--latent", type=int, default=cfg.get("latent", 32))
    ap.add_argument("--beta", type=float, default=cfg.get("beta", 0.25))
    ap.add_argument("--epochs", type=int, default=cfg.get("epochs", 2000))
    ap.add_argument("--batch_size", type=int, default=cfg.get("batch_size", 256))
    ap.add_argument("--lr", type=float, default=cfg.get("lr", 1e-4))
    ap.add_argument("--pretrain_epochs", type=int,
                    default=cfg.get("pretrain_epochs", 200))
    ap.add_argument("--pretrain_lr", type=float,
                    default=cfg.get("pretrain_lr", 1e-3))
    ap.add_argument("--init_n_init", type=int,
                    default=cfg.get("init_n_init", 10))
    ap.add_argument("--min_utilization", type=float,
                    default=cfg.get("min_utilization", 0.9))
    ap.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    ap.add_argument("--log_every", type=int, default=cfg.get("log_every", 100))
    ap.add_argument("--device", default=cfg.get("device", "cpu"))
    ap.add_argument("--checkpoint", default=None,
                    help="Kmeans 保存 .npz 码本;RQ-VAE 保存 .pt 权重与配置")
    ap.add_argument("--no_dedup", action="store_true",
                    help="不追加消歧码,直接用 n_levels 位(跑备选方案用)")
    ap.add_argument("--out", default="semantic_ids.json")
    return ap.parse_args()


def main():
    args = parse_args()

    X = np.load(args.emb).astype(np.float32)
    assert X.ndim == 2 and X.shape[1] == args.in_dim, \
        f"输入形状 {X.shape} 与 config in_dim={args.in_dim} 不一致"
    assert np.isfinite(X).all(), "输入 embedding 含 NaN 或 Inf"
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5), \
        "输入 embedding 不是协议 v3.1 的单位范数向量"
    print("最终配置:", vars(args))
    print(f"输入 {X.shape} | norm mean={np.linalg.norm(X, axis=1).mean():.8f}")

    if args.method == "kmeans":
        codes, codebooks = rq_kmeans(X, args.n_levels, args.codebook_size, args.seed)
        if args.checkpoint:
            Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.checkpoint, codebooks=codebooks,
                     n_levels=args.n_levels, codebook_size=args.codebook_size,
                     in_dim=args.in_dim, seed=args.seed)
            print(f"saved quantizer {args.checkpoint}")
    else:
        model, codes, best = train_rqvae(
            X, args.n_levels, args.codebook_size, args.latent, args.beta,
            args.epochs, args.batch_size, args.lr, args.seed,
            args.log_every, args.device, args.pretrain_epochs,
            args.pretrain_lr, args.init_n_init, args.min_utilization)
        print(f"RQ-VAE 最佳重建 R2 = {best['r2']:.4f}")
        if args.checkpoint:
            Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": {k: v.detach().cpu()
                               for k, v in model.state_dict().items()},
                "config": vars(args),
                "best_epoch": best["epoch"],
                "best_r2": best["r2"],
                "best_utilization": best["util"],
                "best_perplexity": best["ppl"],
                "best_unique_ids": best["unique"],
                "input_shape": list(X.shape),
            }, args.checkpoint)
            print(f"saved checkpoint {args.checkpoint}")

    report(codes, f"{args.method} 消解前", args.codebook_size)

    final = np.asarray(codes) if args.no_dedup else dedup(codes)

    report(final, f"{args.method} 消解后", args.codebook_size,
           has_dedup=not args.no_dedup)
    n_uniq = len({tuple(r) for r in final.tolist()})
    if args.no_dedup:
        if n_uniq < len(final):
            print(f"WARNING: 纯 RQ 结果仍有 {len(final) - n_uniq} 个重复 ID;"
                  "这是备选方案的实测结果,保留并写入报告")
    else:
        assert n_uniq == len(final), f"消解后仍有重复:{len(final) - n_uniq} 个"

    # ★ 容量红线:第 4 位消歧码只有 codebook_size 个位置(256).
    #   消歧码 = 组内序号,所以"最大消歧码 + 1"就是最大冲突组的大小.
    #   超过 256 说明有 257 个以上物品拿到了完全相同的前 3 位 → 编码器塌了.
    if not args.no_dedup:
        max_dedup = int(final[:, -1].max())
        print(f"最大冲突组大小 = {max_dedup + 1}(第 4 位容量 {args.codebook_size})")
        assert max_dedup < args.codebook_size, (
            f"最大冲突组 {max_dedup + 1} 个物品 > 第 4 位容量 {args.codebook_size}.\n"
            f"说明编码器几乎没学到区分性,{args.n_levels} 位前缀大面积撞车.\n"
            f"处理顺序:① 换 bge-m3(1024 维)重跑编码;"
            f"② 检查文本模板有没有把片名丢掉;"
            f"③ 仍不行就启用协议 v3.1 的备选方案(第 4 层 RQ 残差量化).\n"
            f"**不要**直接 --no_dedup 硬跑,那样语义 ID 会重复,第 5 周直接崩.")

    with output_path(args.out).open("w", encoding="utf-8") as f:
        json.dump({str(i): final[i].tolist() for i in range(len(final))}, f)
    print(f"saved {args.out}  形状 {final.shape}")


if __name__ == "__main__":
    main()
