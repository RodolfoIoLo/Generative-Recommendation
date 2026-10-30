"""Resume frozen Stage 7.11 evaluations, skipping verified completed runs."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_week7 import check_gpu
from week7_common import ROOT, artifact_hash, read_json, read_jsonl, require_test_unlock, sha256


def result_tag(name, seed):
    if name in ("sasrec", "gru4rec"):
        return f"{name}_h20_s{seed}_test"
    return f"{name}_s{seed}_test_beam20_n6040"


def completed_result(root, name, seed):
    root = Path(root)
    tag = result_tag(name, seed)
    folder = root / ROOT / "06-evaluation"
    trained_name = (f"{name}_h20_s{seed}" if name in ("sasrec", "gru4rec")
                    else f"{name}_s{seed}")
    try:
        result = read_json(folder / f"{tag}.json")
        metadata_file = trained_name + (".json" if name in ("sasrec", "gru4rec") else "_best.json")
        trained = read_json(root / ROOT / "05-formal-training" / metadata_file)
        rows = read_jsonl(root / "work/week7/predictions" / f"{tag}.jsonl")
        samples = read_json(folder / f"{tag}_samples.json")
        if (result["split"] != "test" or result["n_users"] != 6040
                or result["seed"] != seed or not result["full_sort"]
                or not result["mask_full_history"] or len(rows) != 6040
                or len({int(row["user"]) for row in rows}) != 6040
                or samples != rows[:20]):
            return False
        if any(len(row["ranked_items"]) != 10 or len(set(row["ranked_items"])) != 10
               for row in rows):
            return False
        checkpoint = Path(trained["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = root / checkpoint
        if name in ("sasrec", "gru4rec"):
            return (result["model"].lower() == name and result["max_history"] == 20
                    and result["checkpoint_sha256"] == trained["checkpoint_sha256"]
                    == sha256(checkpoint))
        if (not trained["finished"] or trained["smoke"] or result["variant"] != name
                or result["beam"] != 20 or any(result["diagnostics"].values())
                or result["artifact_hash"] != trained["artifact_hash"]):
            return False
        return bool(result["checkpoint_sha256"]) and all(
            Path(filename).name == filename and sha256(checkpoint / filename) == digest
            for filename, digest in result["checkpoint_sha256"].items()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--data_path")
    parser.add_argument("--kind", choices=("all", "generative", "baseline"), default="all")
    parser.add_argument("--wait", action="store_true", help="Wait for the chosen GPU to become idle.")
    parser.add_argument("--status", action="store_true", help="Verify outputs and list pending jobs only.")
    args = parser.parse_args()
    selection = require_test_unlock("text", 42)
    for name in ("text", selection["selected"]):
        if artifact_hash(f"exported/week7/{name}") != selection["artifact_hashes"][name]:
            raise ValueError(f"frozen evaluation inputs changed: {name}")
    kinds = []
    if args.kind in ("all", "generative"):
        kinds += ["text", selection["selected"]]
    if args.kind in ("all", "baseline"):
        kinds += ["sasrec", "gru4rec"]
    jobs = [(name, seed) for name in kinds for seed in (42, 43, 44)]
    pending = []
    for name, seed in jobs:
        done = completed_result(".", name, seed)
        print(f"{'COMPLETE' if done else 'PENDING'} {name} seed{seed}", flush=True)
        if not done:
            pending.append((name, seed))
    if args.status:
        print(f"{len(jobs) - len(pending)}/{len(jobs)} verified; {len(pending)} pending.")
        return
    if args.gpu is None or args.gpu < 0:
        parser.error("specify a physical --gpu N")
    if args.kind in ("all", "baseline") and not args.data_path:
        parser.error("baseline evaluation needs the RecBole --data_path")
    Path("work/week7").mkdir(parents=True, exist_ok=True)
    # A single coordinator avoids duplicate evaluations from overlapping queues.
    import fcntl
    with Path("work/week7/test-resume.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another resume queue is already running; inspect its tmux session")
        for name, seed in pending:
            if completed_result(".", name, seed):
                continue
            while True:
                try:
                    check_gpu(args.gpu)
                except SystemExit as error:
                    if not args.wait:
                        raise
                    print(f"Waiting for GPU: {error}", flush=True)
                    time.sleep(30)
                    continue
                if shutil.disk_usage(".").free < 128 * 1024**2:
                    if not args.wait:
                        raise SystemExit("Need at least 128 MiB free for evaluation outputs.")
                    print("Waiting for evaluation storage space.", flush=True)
                    time.sleep(30)
                    continue
                if name in ("sasrec", "gru4rec"):
                    command = [sys.executable, "-u", "scripts/week7_baseline.py",
                               "--model", {"sasrec": "SASRec", "gru4rec": "GRU4Rec"}[name],
                               "--seed", str(seed), "--split", "test",
                               "--data_path", args.data_path, "--gpu", str(args.gpu)]
                else:
                    command = [sys.executable, "-u", "scripts/week7_evaluate.py",
                               "--config", f"config/week7/generative_{name}.yaml",
                               "--seed", str(seed), "--split", "test",
                               "--gpu", str(args.gpu), "--beam", "20", "--eval_batch", "8"]
                print("Running:", command, flush=True)
                subprocess.run(command, check=True)
                if not completed_result(".", name, seed):
                    raise RuntimeError(f"incomplete output after evaluation: {name} seed{seed}")
                break
        print("All requested test runs verified. Next: python scripts/run_week7.py summarize")


if __name__ == "__main__":
    main()
