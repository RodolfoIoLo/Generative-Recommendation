"""Week 7 T5 training with deterministic batch-position resume."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

from week7_common import (
    PROTOCOL, ROOT, append_jsonl, artifact_hash, configure_gpu,
    environment, load_config, object_hash, read_json, read_jsonl,
    scheduler_ratio, write_json, write_jsonl,
)


class RestartableBatches:
    """A batch sequence determined only by seed, epoch, and consumed batches."""
    def __init__(self, size, batch_size, seed, start=0):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.start = start
        self.per_epoch = math.ceil(size / batch_size)

    def __iter__(self):
        import torch
        epoch, offset = divmod(self.start, self.per_epoch)
        while True:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            order = torch.randperm(self.size, generator=generator).tolist()
            for batch in range(offset, self.per_epoch):
                yield order[batch * self.batch_size:(batch + 1) * self.batch_size]
            epoch, offset = epoch + 1, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop_after", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--accumulation", type=int)
    parser.add_argument("--eval_batch_size", type=int)
    args = parser.parse_args()
    if args.cpu and not args.smoke:
        parser.error("--cpu is for the small smoke model; formal runs use GPU")
    if args.gpu is None and not args.cpu:
        parser.error("specify physical --gpu N, or --smoke --cpu")
    configure_gpu(args.gpu)

    import torch
    from torch.utils.data import DataLoader
    from transformers import T5Config, T5ForConditionalGeneration
    from generative_common import (
        EvalDataset, SemanticArtifacts, SemanticTrie, TrainCollator,
        TrainPrefixDataset, build_groups, evaluate_model, seed_everything,
    )
    from train_generative import restore_rng, save_checkpoint

    cfg = copy.deepcopy(load_config(args.config))
    cfg["train"]["seed"] = args.seed
    name = cfg["variant"]
    run = f"{name}_s{args.seed}"
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.accumulation is not None:
        cfg["train"]["gradient_accumulation"] = args.accumulation
    if args.eval_batch_size is not None:
        cfg["train"]["eval_batch_size"] = args.eval_batch_size
    if args.smoke:
        run = "smoke_" + run
        cfg["model"].update(d_model=32, d_ff=64, num_layers=1,
                            num_decoder_layers=1, num_heads=2, d_kv=16)
        cfg["train"].update(batch_size=2, gradient_accumulation=1,
                            max_steps=20, warmup_steps=2, eval_steps=2,
                            eval_batch_size=2, early_stop_after=20)
    elif name in ("text", "fixed", "fixed_match", "adaptive") and args.seed != 42:
        selection = read_json(ROOT / "selection.json")
        if name not in ("text", selection["selected"]):
            raise ValueError("seed43/44 is restricted to Text and the frozen winner")
    seed_everything(args.seed)
    frozen = Path(cfg["paths"]["frozen_dir"])
    a = SemanticArtifacts(frozen)
    train_rows = read_jsonl(frozen / "train.semantic.jsonl")
    valid_rows = sorted(read_jsonl(frozen / "valid.semantic.jsonl"),
                        key=lambda row: int(row["user"]))
    if len(a.item_to_code) != 3416 or len(train_rows) != 6040:
        raise ValueError("unexpected frozen artifacts")
    train = TrainPrefixDataset(train_rows, a, 20, 256 if args.smoke else None)
    valid = EvalDataset(valid_rows[:8] if args.smoke else valid_rows, a, 20)
    if not args.smoke and len(train) != 981491:
        raise ValueError("train prefix count changed")
    if not args.smoke and cfg["train"]["batch_size"] * cfg["train"]["gradient_accumulation"] != 256:
        raise ValueError("formal effective batch must remain 256")
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use CPU preparation steps")
    device = torch.device("cpu" if args.cpu else "cuda:0")
    cfg["train"]["bf16"] = (
        device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    groups, _ = build_groups(train_rows, a)
    trie = SemanticTrie(a)
    destination = Path(cfg["paths"]["checkpoint_root"]) / run
    history = ROOT / "05-formal-training" / f"{run}_history.jsonl"
    best_file = ROOT / "05-formal-training" / f"{run}_best.json"
    last = destination / "last"
    identity = {
        "protocol": PROTOCOL, "variant": name, "seed": args.seed,
        "artifact_hash": artifact_hash(frozen), "config": cfg,
        "smoke": args.smoke,
    }
    signature = object_hash(identity)
    if not args.resume and (last.exists() or history.exists() or best_file.exists()):
        raise ValueError(f"run exists: {run}; use --resume")
    if args.resume and not last.exists():
        raise ValueError(f"missing last checkpoint: {last}")
    if args.resume:
        model = T5ForConditionalGeneration.from_pretrained(last).to(device)
    else:
        model = T5ForConditionalGeneration(T5Config(
            **cfg["model"], feed_forward_proj="relu",
        )).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    total = int(cfg["train"]["max_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: scheduler_ratio(
            step, cfg["train"]["warmup_steps"], total,
            cfg["train"]["min_lr_ratio"],
        ),
    )
    step, best, anchor, bad = 0, -1.0, -1.0, 0
    if args.resume:
        state = torch.load(last / "trainer_state.pt",
                           map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise ValueError("resume config/artifact identity mismatch")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, best = state["global_step"], state["best"]
        anchor, bad = state["anchor"], state["bad"]
        restore_rng(state["rng"])
        if history.exists():
            write_jsonl(history, [row for row in read_jsonl(history)
                                 if int(row["step"]) <= step])
    t = cfg["train"]
    loader = DataLoader(
        train,
        batch_sampler=RestartableBatches(
            len(train), t["batch_size"], args.seed,
            start=step * t["gradient_accumulation"],
        ),
        collate_fn=TrainCollator(a.pad), num_workers=0,
        generator=torch.Generator().manual_seed(args.seed + 100000),
    )
    batches = iter(loader)
    limit = min(total, args.stop_after) if args.stop_after is not None else total
    if limit <= step:
        raise ValueError("requested stop is not after the restored step")
    print({"run": run, "device": str(device), "signature": signature,
           "start_step": step, "stop_step": limit}, flush=True)
    last_score = None
    while step < limit:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(t["gradient_accumulation"]):
            batch = {key: value.to(device) for key, value in next(batches).items()}
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16, enabled=t["bf16"]):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            (loss / t["gradient_accumulation"]).backward()
            loss_sum += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["max_grad_norm"])
        optimizer.step()
        scheduler.step()
        step += 1
        if step % t["logging_steps"] == 0 or args.smoke:
            record = {"type": "train", "step": step,
                      "loss": loss_sum / t["gradient_accumulation"],
                      "lr": scheduler.get_last_lr()[0]}
            append_jsonl(history, record)
            print(record, flush=True)
        if step % t["eval_steps"] != 0 and step != limit:
            continue
        result = evaluate_model(model, valid, a, trie, groups, cfg,
                                device, mode="constrained")
        result.pop("predictions")
        score = result["metrics"]["all"]["ndcg@10"]
        last_score = score
        append_jsonl(history, {"type": "valid", "step": step, **result})
        improved = score > best
        if improved:
            best = score
        if step >= t["early_stop_after"]:
            if score >= anchor + t["min_delta"]:
                anchor, bad = score, 0
            else:
                bad += 1
        else:
            anchor, bad = max(anchor, score), 0
        finished = step >= total or bad >= t["patience"]
        state = {"signature": signature, "global_step": step,
                 "best": best, "anchor": anchor, "bad": bad}
        save_checkpoint(last, model, optimizer, scheduler, state)
        if improved:
            save_checkpoint(destination / "best", model, optimizer, scheduler, state)
            write_json(best_file, {
                "protocol_version": PROTOCOL, "variant": name, "seed": args.seed,
                "step": step, "best_step": step, "smoke": args.smoke,
                "finished": finished, "artifact_hash": identity["artifact_hash"],
                "signature": signature, "config": cfg, "metrics": result["metrics"],
                "checkpoint": str(destination / "best"),
                "environment": environment(),
            })
        else:
            record = read_json(best_file)
            record["finished"] = finished
            write_json(best_file, record)
        print({"step": step, "valid_ndcg10": score, "best": best,
               "bad_rounds": bad, "finished": finished}, flush=True)
        if finished:
            break
    print({"run": run, "last_step": step, "last_valid": last_score}, flush=True)


if __name__ == "__main__":
    main()