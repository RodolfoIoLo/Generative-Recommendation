"""RecBole h20 baselines: masked valid checkpoint selection, separate test."""
from __future__ import annotations

import argparse
from pathlib import Path

from week7_common import (
    ROOT, configure_gpu, read_json, read_jsonl, require_test_unlock,
    sha256, write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("SASRec", "GRU4Rec"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    args = parser.parse_args()
    configure_gpu(args.gpu)
    import torch
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.trainer import Trainer
    from recbole.utils import get_model, init_logger, init_seed
    from generative_common import SemanticArtifacts, aggregate_rows, build_groups

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for this baseline training/evaluation")
    a = SemanticArtifacts("exported")
    train_rows = read_jsonl("exported/train.semantic.jsonl")
    valid_rows = read_jsonl("exported/valid.semantic.jsonl")
    groups, _ = build_groups(train_rows, a)
    name = f"{args.model.lower()}_h20_s{args.seed}"
    info_path = ROOT / "05-formal-training" / f"{name}.json"
    cfg = Config(
        model=args.model, dataset="ml-1m",
        config_file_list=["config/sasrec_ml1m.yaml"],
        config_dict={
            "seed": args.seed, "gpu_id": "0", "data_path": args.data_path,
            "MAX_ITEM_LIST_LENGTH": 20, "eval_batch_size": 512,
            "checkpoint_dir": f"saved/week7/baselines/{name}",
            "show_progress": False,
        },
    )
    init_seed(args.seed, True)
    init_logger(cfg)
    dataset = create_dataset(cfg)
    train_data, valid_data, test_data = data_preparation(cfg, dataset)
    model = get_model(args.model)(cfg, train_data._dataset).to(cfg["device"])

    def external(field, inner):
        token = dataset.id2token(field, int(inner))
        if isinstance(token, bytes):
            token = token.decode()
        return int(str(token))

    item_inner = {external(dataset.iid_field, i): i
                  for i in range(1, dataset.item_num)}
    if set(item_inner) != set(a.item_to_code):
        raise ValueError("RecBole item mapping differs from frozen mapping")

    def evaluate(loader, frozen_rows):
        import numpy as np
        by_user = {int(row["user"]): row for row in frozen_rows}
        output = []
        model.eval()
        with torch.inference_mode():
            for interaction, _, positive_u, positive_i in loader:
                interaction = interaction.to(cfg["device"])
                scores = model.full_sort_predict(interaction).view(-1, dataset.item_num)
                scores[:, 0] = -torch.inf
                users = [external(dataset.uid_field, x) for x in
                         interaction[dataset.uid_field].detach().cpu().tolist()]
                targets = {}
                for row_index, inner in zip(positive_u.cpu().tolist(),
                                             positive_i.cpu().tolist()):
                    if row_index in targets:
                        raise ValueError("expected one target per user")
                    targets[row_index] = external(dataset.iid_field, inner)
                for row_index, user in enumerate(users):
                    frozen = by_user[user]
                    target = a.code_item(frozen["target"])
                    if targets[row_index] != target:
                        raise ValueError("RecBole split does not match frozen target")
                    seen = {a.code_item(code) for code in frozen["history"]}
                    scores[row_index, [item_inner[x] for x in seen]] = -torch.inf
                    values = scores[row_index].float().cpu().numpy()
                    # Stable tie break by internal ID; target and mask verified above.
                    inner_top = np.argsort(-values, kind="stable")[:10]
                    ranked = [external(dataset.iid_field, x) for x in inner_top]
                    if len(set(ranked)) != 10 or set(ranked) & seen:
                        raise ValueError("invalid baseline Top-10")
                    output.append({"user": user, "target_item": target,
                                   "ranked_items": ranked})
        if len(output) != len(by_user) or len({r["user"] for r in output}) != len(by_user):
            raise ValueError("missing/duplicate baseline users")
        return aggregate_rows(output, groups)

    class AlignedTrainer(Trainer):
        def _valid_epoch(self, valid_data, show_progress=False):
            result = evaluate(valid_data, valid_rows)["all"]
            metrics = {key: value for key, value in result.items() if key != "n"}
            return metrics["ndcg@10"], metrics

    trainer = AlignedTrainer(cfg, model)
    if args.split == "train":
        if info_path.exists():
            raise ValueError("baseline already trained; inspect its metadata before rerunning")
        score, result = trainer.fit(train_data, valid_data, saved=True, show_progress=False)
        write_json(info_path, {
            "model": args.model, "seed": args.seed, "max_history": 20,
            "selected_on": "valid full-sort with complete frozen history masked",
            "best_valid_score": score, "best_valid_metrics": result,
            "checkpoint": trainer.saved_model_file,
            "checkpoint_sha256": sha256(trainer.saved_model_file),
            "data_path": args.data_path,
        })
        print(read_json(info_path))
    else:
        require_test_unlock("text", 42)
        info = read_json(info_path)
        checkpoint = torch.load(info["checkpoint"], map_location=cfg["device"],
                                weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.load_other_parameter(checkpoint.get("other_parameter"))
        metrics = evaluate(test_data, read_jsonl("exported/test.semantic.jsonl"))
        write_json(ROOT / "06-evaluation" / f"{name}_test.json", {
            **info, "split": "test", "n_users": 6040,
            "full_sort": True, "mask_full_history": True, "metrics": metrics,
        })
        print(metrics["all"])


if __name__ == "__main__":
    main()