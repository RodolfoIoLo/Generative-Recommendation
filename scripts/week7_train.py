"""Week 7 T5 training with deterministic batch-position resume."""
from __future__ import annotations

import argparse
import copy
import errno
import json
import math
from pathlib import Path
import shutil
import time

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


def commit_checkpoint(path, populate):
    """Keep the previous checkpoint until its replacement is fully written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    previous = path.with_name(path.name + ".previous")
    if previous.exists():
        if not path.exists():
            previous.replace(path)
        else:
            raise RuntimeError(f"inspect leftover checkpoint backup: {previous}")
    while True:
        while shutil.disk_usage(path.parent).free < 512 * 1024**2:
            print(f"Waiting for checkpoint space at {path.parent}; "
                  "need at least 512 MiB free. Training state is retained in memory.",
                  flush=True)
            time.sleep(30)
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir()
        try:
            populate(temp)
        except Exception as error:
            shutil.rmtree(temp)
            if (getattr(error, "errno", None) in (errno.ENOSPC, errno.EDQUOT)
                    or "no space left on device" in str(error).lower()):
                print(f"Checkpoint storage is full: {path}; preserving the previous "
                      "checkpoint and retrying in 30 seconds.", flush=True)
                time.sleep(30)
                continue
            raise
        try:
            if path.exists():
                path.replace(previous)
            temp.replace(path)
        except Exception:
            if previous.exists() and not path.exists():
                previous.replace(path)
            raise
        if previous.exists():
            shutil.rmtree(previous)
        return


def save_training_checkpoint(path, model, optimizer, scheduler, state):
    import torch
    from train_generative import rng_state
    saved = dict(state, optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict(), rng=rng_state())

    def populate(folder):
        model.save_pretrained(folder, safe_serialization=True)
        torch.save(saved, folder / "trainer_state.pt")

    commit_checkpoint(path, populate)


def repair_best_checkpoint(destination, best_file, identity, state, history):
    """Reconcile best weights/metadata after a failure following the last save."""
    import torch
    signature = object_hash(identity)
    valid = [row for row in history if row["type"] == "valid"
             and int(row["step"]) <= state["global_step"]]
    if not valid:
        raise ValueError("cannot recover best checkpoint without saved validation history")
    best_row = max(valid, key=lambda row: row["metrics"]["all"]["ndcg@10"])
    score = best_row["metrics"]["all"]["ndcg@10"]
    if not math.isclose(score, state["best"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("checkpoint best score differs from saved validation history")
    best_dir = Path(destination) / "best"
    best_state_file = best_dir / "trainer_state.pt"
    weights_exist = (best_dir / "model.safetensors").is_file()
    best_ok = False
    if best_state_file.exists() and weights_exist:
        best_state = torch.load(best_state_file, map_location="cpu", weights_only=False)
        best_ok = (best_state["signature"] == signature
                   and best_state["global_step"] == best_row["step"])
    if not best_ok:
        if best_row["step"] != state["global_step"]:
            raise ValueError("missing earlier best weights; last cannot reconstruct them")
        last = Path(destination) / "last"
        commit_checkpoint(best_dir, lambda folder: shutil.copytree(last, folder, dirs_exist_ok=True))
    try:
        current = read_json(best_file)
    except (FileNotFoundError, json.JSONDecodeError):
        current = {}
    train = identity["config"]["train"]
    finished = (state["global_step"] >= train["max_steps"]
                or state["bad"] >= train["patience"])
    if (best_ok and current.get("signature") == signature
            and current.get("best_step") == best_row["step"]
            and current.get("metrics") == best_row["metrics"]
            and current.get("finished") == finished):
        return
    write_json(best_file, {
        "protocol_version": identity["protocol"], "variant": identity["variant"],
        "seed": identity["seed"], "step": best_row["step"],
        "best_step": best_row["step"], "smoke": identity["smoke"],
        "finished": finished, "artifact_hash": identity["artifact_hash"],
        "signature": signature, "config": identity["config"],
        "metrics": best_row["metrics"], "checkpoint": str(best_dir),
        "environment": environment(),
    })
    print(f"Recovered best checkpoint record at step {best_row['step']} "
          "from saved validation history and matching weights.", flush=True)


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
    from train_generative import restore_rng

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
        repair_best_checkpoint(destination, best_file, identity, state, read_jsonl(history))
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
        save_training_checkpoint(last, model, optimizer, scheduler, state)
        if improved:
            save_training_checkpoint(destination / "best", model, optimizer, scheduler, state)
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
