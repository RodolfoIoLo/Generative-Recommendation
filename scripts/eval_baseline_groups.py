from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# RecBole 1.2.1 与新 PyTorch 的加载兼容只限定在本脚本进程。
_original_torch_load = torch.load


def _compat_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _compat_load

from recbole.quick_start import load_data_and_model

from generative_common import (
    SemanticArtifacts,
    aggregate_rows,
    build_groups,
    read_jsonl,
)


KNOWN = {
    ("sasrec", 42): {"recall@5": 0.1409, "recall@10": 0.2204,
                     "ndcg@5": 0.0884, "ndcg@10": 0.1141},
    ("sasrec", 43): {"recall@5": 0.1523, "recall@10": 0.2339,
                     "ndcg@5": 0.0968, "ndcg@10": 0.1230},
    ("sasrec", 44): {"recall@5": 0.1526, "recall@10": 0.2250,
                     "ndcg@5": 0.0958, "ndcg@10": 0.1191},
    ("gru4rec", 42): {"recall@5": 0.1861, "recall@10": 0.2657,
                      "ndcg@5": 0.1261, "ndcg@10": 0.1517},
    ("gru4rec", 43): {"recall@5": 0.1836, "recall@10": 0.2624,
                       "ndcg@5": 0.1258, "ndcg@10": 0.1512},
    ("gru4rec", 44): {"recall@5": 0.1796, "recall@10": 0.2613,
                       "ndcg@5": 0.1235, "ndcg@10": 0.1498},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["SASRec", "GRU4Rec"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--frozen_dir", default="exported")
    parser.add_argument(
        "--result_dir",
        default="results/week5-6-generative/05-baseline-evaluation",
    )
    return parser.parse_args()


def external_int(dataset, field, inner_id):
    token = dataset.id2token(field, int(inner_id))
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return int(str(token))


def main():
    args = parse_args()
    artifacts = SemanticArtifacts(args.frozen_dir)
    train_rows = read_jsonl(Path(args.frozen_dir) / "train.semantic.jsonl")
    frozen_rows = read_jsonl(Path(args.frozen_dir) / f"{args.split}.semantic.jsonl")
    frozen_by_user = {int(row["user"]): row for row in frozen_rows}
    groups, _ = build_groups(train_rows, artifacts)

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.checkpoint
    )
    loader = valid_data if args.split == "valid" else test_data
    device = config["device"]
    model.eval()
    rows, legacy_rows = [], []
    item_to_inner = {
        external_int(dataset, dataset.iid_field, inner): inner
        for inner in range(1, dataset.item_num)
    }

    with torch.inference_mode():
        for batched_data in loader:
            interaction, history_index, positive_u, positive_i = batched_data
            interaction = interaction.to(device)
            scores = model.full_sort_predict(interaction)
            scores = scores.view(-1, dataset.item_num)
            scores[:, 0] = -torch.inf
            legacy_top_items = torch.topk(scores, k=10, dim=1).indices.cpu()
            user_inner = interaction[dataset.uid_field].detach().cpu().long()
            batch_users = [
                external_int(dataset, dataset.uid_field, inner)
                for inner in user_inner
            ]
            # Some RecBole versions return history_index=None when loading a
            # checkpoint.  Mask from the frozen split instead so valid uses
            # train history and test uses train+valid history in every run.
            for row_index, user in enumerate(batch_users):
                seen_items = [
                    artifacts.code_item(code)
                    for code in frozen_by_user[user]["history"]
                ]
                seen_inner = [item_to_inner[item] for item in seen_items]
                scores[row_index, seen_inner] = -torch.inf
            aligned_top_items = torch.topk(scores, k=10, dim=1).indices.cpu()

            target_inner = torch.full((scores.size(0),), -1, dtype=torch.long)
            target_inner[positive_u.cpu().long()] = positive_i.cpu().long()

            for row_index in range(scores.size(0)):
                assert target_inner[row_index] >= 0, "leave-one-out batch must have one target per user"
                user = batch_users[row_index]
                recbole_target = external_int(dataset, dataset.iid_field, target_inner[row_index])
                frozen = frozen_by_user[user]
                frozen_target = artifacts.code_item(frozen["target"])
                if recbole_target != frozen_target:
                    raise AssertionError(
                        f"split/ID mismatch user={user}: recbole={recbole_target}, frozen={frozen_target}"
                    )
                ranked = [
                    external_int(dataset, dataset.iid_field, inner)
                    for inner in aligned_top_items[row_index]
                ]
                legacy_ranked = [
                    external_int(dataset, dataset.iid_field, inner)
                    for inner in legacy_top_items[row_index]
                ]
                rows.append({
                    "user": user,
                    "target_item": frozen_target,
                    "ranked_items": ranked,
                })
                legacy_rows.append({
                    "user": user,
                    "target_item": frozen_target,
                    "ranked_items": legacy_ranked,
                })

    assert len(rows) == len(frozen_rows) == 6040
    assert len({row["user"] for row in rows}) == len(rows)
    assert len(legacy_rows) == len(rows)
    metrics = aggregate_rows(rows, groups)
    legacy_metrics = aggregate_rows(legacy_rows, groups)
    output = {
        "model": args.model,
        "seed": args.seed,
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "n_users": len(rows),
        "metrics": metrics,
        "legacy_unmasked_metrics": legacy_metrics,
        "protocol": "metrics masks the complete frozen history; legacy_unmasked_metrics reproduces the original RecBole sequential evaluation",
    }

    expected = KNOWN.get((args.model.lower(), args.seed)) if args.split == "test" else None
    if expected:
        for key, old_value in expected.items():
            new_value = legacy_metrics["all"][key]
            if abs(new_value - old_value) > 1e-4:
                raise AssertionError(
                    f"old-log mismatch {key}: recomputed={new_value:.6f}, old={old_value:.4f}"
                )
        output["old_log_alignment"] = "passed (tolerance=1e-4)"

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    name = f"baseline_{args.model.lower()}_{args.split}_seed{args.seed}_metrics.json"
    (result_dir / name).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
