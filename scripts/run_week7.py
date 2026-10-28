"""Explicit command dispatch; require a chosen idle physical GPU for GPU jobs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from week7_common import ROOT, VARIANTS, read_json


def check_gpu(index):
    command = [
        "nvidia-smi", f"--id={index}",
        "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    memory, utilization = map(int, result.stdout.strip().split(","))
    processes = subprocess.run([
        "nvidia-smi", f"--id={index}", "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, check=True).stdout.strip()
    active = [line for line in processes.splitlines()
              if line.strip() and "No running" not in line]
    if memory > 1024 or utilization > 10 or active:
        raise SystemExit(
            f"GPU {index} is occupied (used={memory} MiB, util={utilization}%, "
            f"compute PIDs={active}). Run CPU steps or retry when idle."
        )
    print(f"GPU {index} appears idle; recheck shared-server availability.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=(
        "init", "features", "tokenize", "curves", "diagnostics", "train", "smoke",
        "evaluate", "exact", "baseline", "select", "summarize", "release", "scaffold",
    ))
    parser.add_argument("--variant", choices=(*VARIANTS, "historical", "selected"),
                        default="text")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop_after", type=int)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--beam", type=int, default=20)
    parser.add_argument("--max_users", type=int)
    parser.add_argument("--candidate_batch", type=int, default=256)
    parser.add_argument("--eval_batch", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--checkpoint")
    parser.add_argument("--model", choices=("SASRec", "GRU4Rec"))
    parser.add_argument("--data_path")
    parser.add_argument("--author")
    parser.add_argument("--url")
    parser.add_argument("--license", choices=("MIT",))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    variant = args.variant
    if variant == "selected":
        variant = read_json(ROOT / "selection.json")["selected"]
    cfg = f"config/week7/generative_{variant}.yaml"
    command = [sys.executable]
    gpu_job = args.action in ("train", "evaluate", "exact", "baseline")
    if gpu_job and args.gpu is None:
        parser.error(f"{args.action} requires a physical --gpu N")
    if args.action in ("init", "features", "tokenize"):
        command += ["scripts/week7_prepare.py", args.action]
        if args.action == "tokenize":
            command += ["--variant", variant]
    elif args.action in ("curves", "diagnostics", "select", "summarize"):
        command += ["scripts/week7_report.py", args.action]
    elif args.action in ("train", "smoke"):
        command += ["scripts/week7_train.py", "--config", cfg,
                    "--seed", str(args.seed)]
        if args.action == "smoke":
            command += ["--smoke", "--cpu"]
        else:
            command += ["--gpu", str(args.gpu),
                        "--batch_size", str(args.batch_size),
                        "--accumulation", str(args.accumulation),
                        "--eval_batch_size", str(args.eval_batch)]
        if args.resume:
            command += ["--resume"]
        if args.stop_after is not None:
            command += ["--stop_after", str(args.stop_after)]
    elif args.action in ("evaluate", "exact"):
        if args.split == "train":
            parser.error("evaluate/exact requires valid or test")
        command += [
            "scripts/week7_evaluate.py", "--config", cfg,
            "--seed", str(args.seed), "--gpu", str(args.gpu),
            "--split", args.split, "--beam", str(args.beam),
            "--eval_batch", str(args.eval_batch),
            "--candidate_batch", str(args.candidate_batch),
        ]
        if args.max_users is not None:
            command += ["--max_users", str(args.max_users)]
        if args.action == "exact":
            command += ["--exact"]
        if args.checkpoint:
            command += ["--checkpoint", args.checkpoint]
        if variant == "historical" and args.checkpoint:
            command += ["--historical"]
    elif args.action == "baseline":
        if args.model is None or args.data_path is None or args.split == "valid":
            parser.error("baseline needs --model, --data_path, --split train|test")
        command += [
            "scripts/week7_baseline.py", "--model", args.model,
            "--seed", str(args.seed), "--gpu", str(args.gpu),
            "--data_path", args.data_path, "--split", args.split,
        ]
    else:
        if not all((args.author, args.url, args.license)):
            parser.error("release needs --author, --url, --license MIT")
        command += [
            "scripts/week7_release.py", "--author", args.author,
            "--url", args.url, "--license", args.license,
        ]
        if args.action == "scaffold":
            command += ["--scaffold"]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if gpu_job:
        check_gpu(args.gpu)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()