#!/usr/bin/env python
"""跑判别式基线(SASRec / GRU4Rec).

用法:
    python scripts/week2_train_baseline.py --model SASRec --seed 43
    python scripts/week2_train_baseline.py --model GRU4Rec --seed 42
"""
import argparse
import functools
from logging import getLogger

import torch

_original_torch_load = torch.load


@functools.wraps(_original_torch_load)
def _torch_load_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false
# RecBole checkpoints include configuration objects, requiring a full trusted load.

from recbole.quick_start import run_recbole  # noqa: E402
from recbole.utils import dict2str  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="SASRec")
    ap.add_argument("--dataset", default="ml-1m")
    ap.add_argument("--config", default="config/sasrec_ml1m.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_id", default="0")
    ap.add_argument("--data_path", help="Parent of the ml-1m RecBole dataset directory")
    args = ap.parse_args()

    overrides = {"seed": args.seed, "gpu_id": args.gpu_id}
    if args.data_path is not None:
        overrides["data_path"] = args.data_path

    result = run_recbole(
        model=args.model,
        dataset=args.dataset,
        config_file_list=[args.config],
        config_dict=overrides,
    )

    logger = getLogger()
    logger.info("test result:\n%s", dict2str(result["test_result"]))
    print("\n===== TEST RESULT =====")
    print(dict2str(result["test_result"]))


if __name__ == "__main__":
    main()
