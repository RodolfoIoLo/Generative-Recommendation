"""Beam / exact catalog diagnostics and frozen formal evaluation."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import time

from week7_common import (
    PROTOCOL, ROOT, artifact_hash, configure_gpu, load_config, read_json,
    read_jsonl, require_test_unlock, sha256, write_json, write_jsonl,
)


def exact_rows(model, rows, artifacts, device, candidate_batch):
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    output = []
    catalog = sorted(artifacts.item_to_code)
    with torch.inference_mode():
        for index, row in enumerate(rows):
            seen = {artifacts.code_item(code) for code in row["history"]}
            candidates = [item for item in catalog if item not in seen]
            source = artifacts.encode_items(row["history"], 20)
            inputs = torch.tensor([source], dtype=torch.long, device=device)
            mask = torch.ones_like(inputs)
            encoded = model.get_encoder()(input_ids=inputs,
                                          attention_mask=mask).last_hidden_state
            scores = []
            for start in range(0, len(candidates), candidate_batch):
                items = candidates[start:start + candidate_batch]
                labels = torch.tensor([
                    artifacts.code_to_tokens(artifacts.item_to_code[item])
                    + [artifacts.eos] for item in items
                ], dtype=torch.long, device=device)
                logits = model(
                    encoder_outputs=BaseModelOutput(
                        last_hidden_state=encoded.expand(len(items), -1, -1)),
                    attention_mask=mask.expand(len(items), -1),
                    labels=labels, use_cache=False,
                ).logits
                log_prob = logits.float().log_softmax(-1)
                values = log_prob.gather(-1, labels.unsqueeze(-1)).squeeze(-1).sum(-1)
                scores.extend(values.cpu().tolist())
            # Stable tie break is the ascending item ID order.
            ranking = sorted(range(len(candidates)), key=lambda j: (-scores[j], candidates[j]))
            chosen = ranking[:10]
            output.append({
                "user": int(row["user"]),
                "target_item": artifacts.code_item(row["target"]),
                "ranked_items": [candidates[j] for j in chosen],
                "model_log_prob": [scores[j] for j in chosen],
            })
            if (index + 1) % 32 == 0:
                print(f"exact users: {index + 1}/{len(rows)}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--beam", type=int, default=20)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--max_users", type=int)
    parser.add_argument("--candidate_batch", type=int, default=256)
    parser.add_argument("--eval_batch", type=int, default=8)
    parser.add_argument("--checkpoint")
    parser.add_argument("--historical", action="store_true")
    args = parser.parse_args()
    configure_gpu(args.gpu)
    import torch
    from transformers import T5ForConditionalGeneration
    from generative_common import (
        EvalDataset, SemanticArtifacts, SemanticTrie, aggregate_rows,
        build_groups, evaluate_model, seed_everything,
    )
    cfg = copy.deepcopy(load_config(args.config))
    variant = cfg["variant"]
    if args.split == "test":
        if args.exact or args.max_users or args.historical or args.beam != 20:
            parser.error("test must use full users, beam20, and the frozen formal model")
        require_test_unlock(variant, args.seed)
    if args.beam < 10:
        parser.error("beam must be >= Top-10")
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA GPU; exact/beam experiments can wait")
    frozen = Path(cfg["paths"]["frozen_dir"])
    a = SemanticArtifacts(frozen)
    train = read_jsonl(frozen / "train.semantic.jsonl")
    rows = sorted(read_jsonl(frozen / f"{args.split}.semantic.jsonl"),
                  key=lambda row: int(row["user"]))
    if args.max_users:
        rows = rows[:args.max_users]
    if args.historical:
        if not args.checkpoint:
            parser.error("--historical requires --checkpoint")
        checkpoint = Path(args.checkpoint)
        run_name = "historical_30k_s42"
    else:
        run_name = f"{variant}_s{args.seed}"
        meta = read_json(ROOT / "05-formal-training" / f"{run_name}_best.json")
        if not meta["finished"] or meta["artifact_hash"] != artifact_hash(frozen):
            raise ValueError("unfinished run or changed artifacts")
        checkpoint = Path(meta["checkpoint"])
    cfg["decode"]["num_beams"] = args.beam
    cfg["train"].update(eval_batch_size=args.eval_batch, num_workers=0)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint).to("cuda:0").eval()
    model.generation_config.renormalize_logits = False
    model.generation_config.length_penalty = 0.0
    groups, _ = build_groups(train, a)
    warm_source = torch.tensor([a.encode_items(rows[0]["history"], 20)],
                               dtype=torch.long, device="cuda:0")
    warm_label = torch.tensor([a.code_to_tokens(rows[0]["target"]) + [a.eos]],
                              dtype=torch.long, device="cuda:0")
    with torch.inference_mode():
        model(input_ids=warm_source, labels=warm_label)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    if args.exact:
        predictions = exact_rows(model, rows, a, "cuda:0", args.candidate_batch)
        result = {
            "metrics": aggregate_rows(predictions, groups),
            "diagnostics": {"catalog_invalid_rate": 0.0,
                            "seen_repeat_rate": 0.0, "short_list_rate": 0.0},
        }
    else:
        result = evaluate_model(
            model, EvalDataset(rows, a, 20), a, SemanticTrie(a), groups,
            cfg, torch.device("cuda:0"), mode="constrained",
        )
        predictions = result.pop("predictions")
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    for row, source in zip(predictions, rows):
        seen = {a.code_item(code) for code in source["history"]}
        ranked = row["ranked_items"]
        if len(set(ranked)) != 10 or None in ranked or set(ranked) & seen:
            raise ValueError("illegal, duplicate, seen, or short Top-10")
    full = args.split == "test" or (
        args.max_users is None and not args.historical and not args.exact)
    folder = ROOT / ("06-evaluation" if full else "01-diagnostics")
    tag = "exact" if args.exact else f"beam{args.beam}"
    filename = f"{run_name}_{args.split}_{tag}_n{len(rows)}"
    weight_files = sorted(checkpoint.glob("*.safetensors")) or sorted(checkpoint.glob("pytorch_model*.bin"))
    if not weight_files:
        raise FileNotFoundError("no model weight file")
    result.update({
        "protocol_version": PROTOCOL, "variant": variant, "seed": args.seed,
        "split": args.split, "n_users": len(rows), "full_sort": True,
        "mask_full_history": True, "beam": None if args.exact else args.beam,
        "scoring": "sum of raw token log probabilities; EOS included",
        "artifact_hash": artifact_hash(frozen),
        "checkpoint_sha256": {file.name: sha256(file) for file in weight_files},
        "seconds": seconds, "milliseconds_per_user": seconds * 1000 / len(rows),
        "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
        "user_ids": [int(row["user"]) for row in rows],
    })
    write_json(folder / f"{filename}.json", result)
    write_jsonl(Path("work/week7/predictions") / f"{filename}.jsonl", predictions)
    write_json(folder / f"{filename}_samples.json", predictions[:20])
    print({"saved": str(folder / f"{filename}.json"),
           "ndcg@10": result["metrics"]["all"]["ndcg@10"]}, flush=True)


if __name__ == "__main__":
    main()