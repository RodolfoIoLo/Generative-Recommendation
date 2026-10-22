from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration

from generative_common import (
    EvalDataset,
    SemanticArtifacts,
    SemanticTrie,
    build_groups,
    evaluate_model,
    load_yaml,
    read_jsonl,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--split", choices=["valid", "test"], required=True)
    parser.add_argument(
        "--mode", choices=["constrained", "unconstrained", "both"],
        default="both",
    )
    parser.add_argument("--max_users", type=int)
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("full beam-search evaluation requires CUDA")

    frozen = Path(cfg["paths"]["frozen_dir"])
    artifacts = SemanticArtifacts(frozen)
    train_rows = read_jsonl(frozen / "train.semantic.jsonl")
    eval_rows = read_jsonl(frozen / f"{args.split}.semantic.jsonl")
    groups, _ = build_groups(train_rows, artifacts)
    dataset = EvalDataset(
        eval_rows, artifacts, int(cfg["data"]["max_history_items"]),
        args.max_users,
    )
    trie = SemanticTrie(artifacts)
    device = torch.device("cuda:0")
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint).to(device)
    model.eval()

    result_dir = Path(cfg["paths"]["result_root"])
    result_dir.mkdir(parents=True, exist_ok=True)
    modes = ["unconstrained", "constrained"] if args.mode == "both" else [args.mode]
    for mode in modes:
        result = evaluate_model(
            model, dataset, artifacts, trie, groups, cfg, device, mode=mode,
        )
        predictions = result.pop("predictions")
        result.update({
            "model": "Generative-T5",
            "seed": args.seed,
            "split": args.split,
            "checkpoint": str(Path(args.checkpoint).resolve()),
        })
        stem = f"generative_{args.split}_{mode}_seed{args.seed}"
        metrics_path = result_dir / f"{stem}_metrics.json"
        metrics_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if args.save_predictions:
            write_jsonl(result_dir / f"{stem}_predictions.jsonl", predictions)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

        if mode == "constrained":
            diagnostics = result["diagnostics"]
            assert diagnostics["catalog_invalid_rate"] == 0.0
            assert diagnostics["seen_repeat_rate"] == 0.0


if __name__ == "__main__":
    main()