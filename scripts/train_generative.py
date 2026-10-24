from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

from generative_common import (
    SemanticArtifacts,
    SemanticTrie,
    TrainCollator,
    TrainPrefixDataset,
    EvalDataset,
    build_groups,
    evaluate_model,
    load_yaml,
    read_jsonl,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run_name")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--max_train_examples", type=int)
    parser.add_argument("--max_eval_users", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--result_dir")
    parser.add_argument("--audit_only", action="store_true")
    return parser.parse_args()


def build_model(cfg):
    m = cfg["model"]
    model_cfg = T5Config(
        vocab_size=int(m["vocab_size"]),
        d_model=int(m["d_model"]),
        d_ff=int(m["d_ff"]),
        num_layers=int(m["num_layers"]),
        num_decoder_layers=int(m["num_decoder_layers"]),
        num_heads=int(m["num_heads"]),
        d_kv=int(m["d_kv"]),
        dropout_rate=float(m["dropout_rate"]),
        pad_token_id=int(m["pad_token_id"]),
        decoder_start_token_id=int(m["decoder_start_token_id"]),
        eos_token_id=int(m["eos_token_id"]),
        feed_forward_proj="relu",
    )
    return T5ForConditionalGeneration(model_cfg)


def apply_overrides(cfg, args):
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.eval_steps is not None:
        cfg["train"]["eval_steps"] = args.eval_steps
    return cfg


def git_revision():
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ) != 0
        return revision + ("-dirty" if dirty else "")
    except Exception:
        return "uncommitted"


def audit(cfg, artifacts, train_rows, valid_rows, test_rows, model):
    expected = cfg["data"]
    assert len(artifacts.item_to_code) == int(expected["expected_items"])
    assert artifacts.vocab_size == int(cfg["model"]["vocab_size"])
    assert len(valid_rows) == len(test_rows) == int(expected["expected_users"])
    assert len(train_rows) == int(expected["expected_users"])
    train_examples = sum(len(row["items"]) - 1 for row in train_rows)
    assert train_examples == int(expected["expected_train_examples"])
    assert len(artifacts.code_to_item) == len(artifacts.item_to_code)

    def check_code(code):
        assert len(code) == artifacts.length
        assert all(0 <= int(x) < artifacts.k for x in code)
        tokens = artifacts.code_to_tokens(code)
        assert artifacts.tokens_to_code(tokens) == tuple(code)

    for row in train_rows:
        for code in row["items"]:
            check_code(code)
    for rows in (valid_rows, test_rows):
        for row in rows:
            for code in row["history"]:
                check_code(code)
            check_code(row["target"])

    groups, freq = build_groups(train_rows, artifacts)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "items": len(artifacts.item_to_code),
        "users": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "train_interactions": sum(len(row["items"]) for row in train_rows),
        "train_examples": train_examples,
        "vocab_size": artifacts.vocab_size,
        "max_input_tokens": int(cfg["data"]["max_history_items"]) * artifacts.length + 1,
        "parameters": n_params,
        "groups": {name: len(items) for name, items in groups.items()},
        "strict_cold_targets": {
            "valid": sum(artifacts.code_item(row["target"]) in groups["strict_cold"] for row in valid_rows),
            "test": sum(artifacts.code_item(row["target"]) in groups["strict_cold"] for row in test_rows),
        },
        "train_frequency_min_max": [min(freq.values()), max(freq.values())],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, scheduler, state):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    model.save_pretrained(temp, safe_serialization=True)
    state = dict(state)
    state.update({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": rng_state(),
    })
    torch.save(state, temp / "trainer_state.pt")
    if path.exists():
        shutil.rmtree(path)
    temp.replace(path)


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    cfg = apply_overrides(load_yaml(args.config), args)
    frozen = Path(cfg["paths"]["frozen_dir"])
    artifacts = SemanticArtifacts(frozen)
    train_rows = read_jsonl(frozen / "train.semantic.jsonl")
    valid_rows = read_jsonl(frozen / "valid.semantic.jsonl")
    test_rows = read_jsonl(frozen / "test.semantic.jsonl")
    seed = int(cfg["train"]["seed"])
    seed_everything(seed)

    model = build_model(cfg) if not args.resume else T5ForConditionalGeneration.from_pretrained(args.resume)
    audit(cfg, artifacts, train_rows, valid_rows, test_rows, model)
    if args.audit_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("formal training requires an available CUDA GPU")

    run_name = args.run_name or f"t5_sid_seed{seed}"
    checkpoint_dir = Path(cfg["paths"]["checkpoint_root"]) / run_name
    result_dir = Path(
        args.result_dir
        or cfg["paths"].get("training_result_dir", cfg["paths"]["result_root"])
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    history_path = result_dir / f"{run_name}_history.jsonl"

    train_set = TrainPrefixDataset(
        train_rows, artifacts, int(cfg["data"]["max_history_items"]),
        args.max_train_examples,
    )
    valid_set = EvalDataset(
        valid_rows, artifacts, int(cfg["data"]["max_history_items"]),
        args.max_eval_users,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_set,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
        collate_fn=TrainCollator(artifacts.pad),
    )
    groups, _ = build_groups(train_rows, artifacts)
    trie = SemanticTrie(artifacts)

    device = torch.device("cuda:0")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    max_steps = int(cfg["train"]["max_steps"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(int(cfg["train"]["warmup_steps"]), max_steps // 2),
        num_training_steps=max_steps,
    )

    step, best, bad_rounds = 0, float("-inf"), 0
    if args.resume:
        state = torch.load(Path(args.resume) / "trainer_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = int(state["global_step"])
        best = float(state["best_ndcg10"])
        bad_rounds = int(state["bad_rounds"])
        restore_rng(state["rng"])
        print(f"resumed from optimizer step {step}", flush=True)

    print(json.dumps({
        "run_name": run_name,
        "seed": seed,
        "git": git_revision(),
        "config": cfg,
        "train_examples": len(train_set),
        "valid_users": len(valid_set),
    }, ensure_ascii=False), flush=True)

    accumulation = int(cfg["train"]["gradient_accumulation"])
    log_every = int(cfg["train"]["logging_steps"])
    eval_every = int(cfg["train"]["eval_steps"])
    use_bf16 = bool(cfg["train"]["bf16"])
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    loss_sum, loss_count = 0.0, 0

    while step < max_steps:
        model.train()
        for _ in range(accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            context = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
            with context:
                raw_loss = model(**batch).loss
                loss = raw_loss / accumulation
            if not torch.isfinite(raw_loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {raw_loss.item()}")
            loss.backward()
            loss_sum += float(raw_loss.detach())
            loss_count += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["max_grad_norm"]))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % log_every == 0:
            record = {
                "type": "train",
                "step": step,
                "loss": loss_sum / max(1, loss_count),
                "lr": scheduler.get_last_lr()[0],
            }
            print(json.dumps(record), flush=True)
            append_jsonl(history_path, record)
            loss_sum, loss_count = 0.0, 0

        if step % eval_every == 0 or step == max_steps:
            evaluated = evaluate_model(
                model, valid_set, artifacts, trie, groups, cfg, device,
                mode="constrained",
            )
            metrics_only = {key: value for key, value in evaluated.items() if key != "predictions"}
            record = {"type": "valid", "step": step, **metrics_only}
            print(json.dumps(record, ensure_ascii=False), flush=True)
            append_jsonl(history_path, record)
            score = evaluated["metrics"]["all"]["ndcg@10"]
            improved = score > best
            if improved:
                best, bad_rounds = score, 0
            else:
                bad_rounds += 1
            state = {
                "global_step": step,
                "best_ndcg10": best,
                "bad_rounds": bad_rounds,
                "seed": seed,
                "run_name": run_name,
                "config": cfg,
            }
            save_checkpoint(checkpoint_dir / "last", model, optimizer, scheduler, state)
            if improved:
                save_checkpoint(checkpoint_dir / "best", model, optimizer, scheduler, state)
                summary_path = result_dir / f"{run_name}_best_valid_metrics.json"
                summary_path.write_text(
                    json.dumps({"step": step, **metrics_only}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            if bad_rounds >= int(cfg["train"]["patience"]):
                print(f"early stop at step {step}; best NDCG@10={best:.6f}", flush=True)
                break


if __name__ == "__main__":
    main()
