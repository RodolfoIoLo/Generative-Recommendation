from __future__ import annotations

import bisect
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SemanticArtifacts:
    def __init__(self, frozen_dir):
        root = Path(frozen_dir)
        self.vocab = json.loads((root / "token_vocab.json").read_text(encoding="utf-8"))
        raw = json.loads((root / "semantic_ids.json").read_text(encoding="utf-8"))
        self.item_to_code = {int(k): tuple(v) for k, v in raw.items()}
        self.code_to_item = {v: k for k, v in self.item_to_code.items()}
        if len(self.code_to_item) != len(self.item_to_code):
            raise ValueError("semantic IDs are not unique")
        self.pad, self.bos, self.eos = 0, 1, 2
        self.k = int(self.vocab["codebook_size"])
        self.length = int(self.vocab["semantic_id_length"])
        self.vocab_size = int(self.vocab["vocab_size"])

    def code_to_tokens(self, code):
        if len(code) != self.length or any(not 0 <= int(x) < self.k for x in code):
            raise ValueError(f"bad semantic code: {code}")
        return [3 + level * self.k + int(value) for level, value in enumerate(code)]

    def tokens_to_code(self, tokens):
        if len(tokens) != self.length:
            return None
        code = []
        for level, token in enumerate(tokens):
            value = int(token) - 3 - level * self.k
            if not 0 <= value < self.k:
                return None
            code.append(value)
        return tuple(code)

    def code_item(self, code):
        return self.code_to_item.get(tuple(code))

    def encode_items(self, codes, max_items):
        codes = codes[-max_items:]
        return [t for code in codes for t in self.code_to_tokens(code)] + [self.eos]


class TrainPrefixDataset(Dataset):
    def __init__(self, rows, artifacts, max_history_items, limit=None):
        self.rows = rows
        self.a = artifacts
        self.max_history_items = max_history_items
        counts = [max(0, len(row["items"]) - 1) for row in rows]
        self.ends = np.cumsum(counts).tolist()
        self.total = self.ends[-1] if self.ends else 0
        if limit is not None:
            self.total = min(self.total, int(limit))

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        row_idx = bisect.bisect_right(self.ends, index)
        previous = 0 if row_idx == 0 else self.ends[row_idx - 1]
        target_pos = index - previous + 1
        items = self.rows[row_idx]["items"]
        source = self.a.encode_items(items[:target_pos], self.max_history_items)
        labels = self.a.code_to_tokens(items[target_pos]) + [self.a.eos]
        return {"input_ids": source, "labels": labels}


class EvalDataset(Dataset):
    def __init__(self, rows, artifacts, max_history_items, limit=None):
        self.rows = rows if limit is None else rows[: int(limit)]
        self.a = artifacts
        self.max_history_items = max_history_items

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        history = [tuple(x) for x in row["history"]]
        return {
            "user": int(row["user"]),
            "input_ids": self.a.encode_items(history, self.max_history_items),
            "history_codes": history,
            "target_code": tuple(row["target"]),
        }


class TrainCollator:
    def __init__(self, pad=0):
        self.pad = pad

    def __call__(self, batch):
        max_x = max(len(x["input_ids"]) for x in batch)
        max_y = max(len(x["labels"]) for x in batch)
        ids, masks, labels = [], [], []
        for row in batch:
            x, y = row["input_ids"], row["labels"]
            ids.append(x + [self.pad] * (max_x - len(x)))
            masks.append([1] * len(x) + [0] * (max_x - len(x)))
            labels.append(y + [-100] * (max_y - len(y)))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class EvalCollator:
    def __init__(self, pad=0):
        self.pad = pad

    def __call__(self, batch):
        max_x = max(len(x["input_ids"]) for x in batch)
        ids, masks = [], []
        for row in batch:
            x = row["input_ids"]
            ids.append(x + [self.pad] * (max_x - len(x)))
            masks.append([1] * len(x) + [0] * (max_x - len(x)))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "users": [x["user"] for x in batch],
            "histories": [x["history_codes"] for x in batch],
            "targets": [x["target_code"] for x in batch],
        }


class SemanticTrie:
    def __init__(self, artifacts):
        self.a = artifacts
        self.children = {(): set()}
        self.descendants = {}
        self.leaves = {}
        for item, code in artifacts.item_to_code.items():
            path = tuple(artifacts.code_to_tokens(code))
            for length in range(len(path) + 1):
                prefix = path[:length]
                self.descendants.setdefault(prefix, set()).add(item)
                if length < len(path):
                    self.children.setdefault(prefix, set()).add(path[length])
            self.children[path] = {artifacts.eos}
            self.leaves[path] = item

    def allowed(self, prefix, seen_items=None):
        prefix = tuple(prefix)
        if seen_items and len(prefix) < self.a.length:
            allowed = []
            for token in self.children.get(prefix, ()):
                child = prefix + (token,)
                if self.descendants.get(child, set()).difference(seen_items):
                    allowed.append(token)
            return sorted(allowed)
        if len(prefix) == self.a.length:
            item = self.leaves.get(prefix)
            return [self.a.eos] if item is not None and item not in (seen_items or set()) else []
        return sorted(self.children.get(prefix, ()))


def rank_metrics(ranked_items, target, ks=(5, 10)):
    out = {}
    for k in ks:
        try:
            rank = ranked_items[:k].index(target) + 1
        except ValueError:
            rank = None
        out[f"recall@{k}"] = float(rank is not None)
        out[f"ndcg@{k}"] = 0.0 if rank is None else 1.0 / math.log2(rank + 1)
    return out


def build_groups(train_rows, artifacts):
    freq = Counter()
    for row in train_rows:
        freq.update(artifacts.code_item(code) for code in row["items"])
    item_ids = sorted(artifacts.item_to_code)
    ordered = sorted(item_ids, key=lambda item: (freq[item], item))
    cut = math.ceil(len(ordered) * 0.8)
    return {
        "strict_cold": {x for x in item_ids if freq[x] == 0},
        "few_shot_le20": {x for x in item_ids if freq[x] <= 20},
        "tail_bottom80": set(ordered[:cut]),
        "head_top20": set(ordered[cut:]),
    }, freq


def decode_generated(sequence, artifacts):
    tokens = [int(x) for x in sequence]
    if tokens and tokens[0] == artifacts.bos:
        tokens = tokens[1:]
    if artifacts.eos not in tokens:
        return None, None
    eos_at = tokens.index(artifacts.eos)
    code = artifacts.tokens_to_code(tokens[:eos_at])
    if code is None:
        return None, None
    return code, artifacts.code_item(code)


def aggregate_rows(rows, groups):
    subsets = {"all": rows}
    for name, members in groups.items():
        subsets[name] = [row for row in rows if row["target_item"] in members]
    output = {}
    for name, selected in subsets.items():
        sums = {"recall@5": 0.0, "recall@10": 0.0,
                "ndcg@5": 0.0, "ndcg@10": 0.0}
        for row in selected:
            current = rank_metrics(row["ranked_items"], row["target_item"])
            for key, value in current.items():
                sums[key] += value
        n = len(selected)
        output[name] = {"n": n}
        output[name].update({key: (value / n if n else None)
                             for key, value in sums.items()})
    return output


def evaluate_model(model, dataset, artifacts, trie, groups, cfg, device,
                   mode="constrained"):
    if mode not in {"constrained", "unconstrained"}:
        raise ValueError(mode)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=EvalCollator(artifacts.pad),
    )
    topk = int(cfg["decode"]["topk"])
    rows, invalid, seen_repeat, short = [], 0, 0, 0
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            histories = batch["histories"]
            seen_sets = [
                {artifacts.code_item(code) for code in history}
                for history in histories
            ]

            prefix_fn = None
            if mode == "constrained":
                def prefix_fn(batch_id, generated_ids):
                    prefix = generated_ids.tolist()
                    if prefix and prefix[0] == artifacts.bos:
                        prefix = prefix[1:]
                    allowed = trie.allowed(prefix, seen_sets[batch_id])
                    if not allowed:
                        raise RuntimeError(
                            f"empty legal continuation batch={batch_id} prefix={prefix}"
                        )
                    return allowed

            generated = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                num_beams=int(cfg["decode"]["num_beams"]),
                num_return_sequences=topk,
                max_new_tokens=int(cfg["decode"]["max_new_tokens"]),
                early_stopping=True,
                prefix_allowed_tokens_fn=prefix_fn,
            )
            generated = generated.detach().cpu().view(len(histories), topk, -1)

            for local in range(len(histories)):
                ranked, codes, used = [], [], set()
                for sequence in generated[local]:
                    code, item = decode_generated(sequence.tolist(), artifacts)
                    codes.append(None if code is None else list(code))
                    if item is None:
                        invalid += 1
                        ranked.append(None)
                    elif item in seen_sets[local]:
                        seen_repeat += 1
                        ranked.append(None)
                    elif item in used:
                        ranked.append(None)
                    else:
                        used.add(item)
                        ranked.append(item)
                if len(used) < topk:
                    short += 1
                target_item = artifacts.code_item(batch["targets"][local])
                if target_item is None:
                    raise ValueError("evaluation target is absent from catalog")
                rows.append({
                    "user": batch["users"][local],
                    "target_item": target_item,
                    "target_code": list(batch["targets"][local]),
                    "ranked_items": ranked,
                    "generated_codes": codes,
                })
            if (batch_index + 1) % 100 == 0:
                print(f"evaluated {len(rows)}/{len(dataset)} users", flush=True)

    denominator = max(1, len(rows) * topk)
    return {
        "mode": mode,
        "n_users": len(rows),
        "metrics": aggregate_rows(rows, groups),
        "diagnostics": {
            "catalog_invalid_rate": invalid / denominator,
            "seen_repeat_rate": seen_repeat / denominator,
            "short_list_rate": short / max(1, len(rows)),
        },
        "predictions": rows,
    }