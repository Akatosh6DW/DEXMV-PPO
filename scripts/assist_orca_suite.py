"""Run a disjoint, explicitly selected tail of the frozen matrix concurrently.

The original runner later verifies and skips completed jobs. Never select jobs
near its active position or launch another assistant over the same range.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--phase", choices=("train", "eval"), required=True)
    p.add_argument("--min-seed", type=int, default=1006)
    p.add_argument("--benchmarks", nargs="+", default=[])
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    if not 1 <= args.workers <= 4:
        p.error("At most four workers")
    if args.workers == 4:
        for entry in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                command = entry.read_bytes().split(b"\0")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if len(command) > 1 and Path(os.fsdecode(command[1])).name == "run_orca_suite_job.py":
                p.error("Four-worker mode requires all previous suite jobs to have drained")
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    for path, expected in protocol["source_hashes"].items():
        assert digest(ROOT / path) == expected, path
    selected = []
    for record in protocol["jobs"]:
        if not record["name"].startswith(args.phase + "/"):
            continue
        path = ROOT / record["path"]
        assert digest(path) == record["sha256"]
        job = json.loads(path.read_text())
        if args.phase == "train" and job["seed"] < args.min_seed:
            continue
        if args.phase == "eval" and job["benchmark"] not in args.benchmarks:
            continue
        if args.seeds is not None and job["seed"] not in args.seeds:
            continue
        out = ROOT / job["output"]
        if out.exists() and not (out / "summary.json").exists():
            raise FileExistsError("Range overlaps a partial/live job: " + str(out))
        if job.get("checkpoint") and not (ROOT / job["checkpoint"]).exists():
            raise FileNotFoundError("Wait for the training phase: " + job["checkpoint"])
        if job.get("checkpoint"):
            parent_summary = ROOT / Path(job["checkpoint"]).parent / "summary.json"
            if not parent_summary.exists():
                raise RuntimeError("Checkpoint is not final yet: " + job["checkpoint"])
            prior = json.loads(parent_summary.read_text())
            assert prior["status"] == "completed" and prior["final_step"] == protocol["training_steps"]
        selected.append((record, path, job))
    if not selected:
        raise ValueError("Empty explicit selection")
    stamp = time.time_ns()
    manifest = root / f"assist_{args.phase}_{stamp}.json"
    manifest.write_text(json.dumps(dict(
        jobs=[record["name"] for record, _, _ in selected], workers=args.workers,
        protocol_sha256=digest(root / "protocol.json"), runner_sha256=digest(Path(__file__)),
        started_epoch=time.time(), rationale="Disjoint tail acceleration, no changes to frozen experiments."), indent=2)+"\n")

    def run(item):
        record, path, job = item
        out = ROOT / job["output"]
        command = [sys.executable, str(ROOT / "scripts/run_orca_suite_job.py"), "--job", str(path)]
        if not (out / "summary.json").exists():
            if out.exists():
                raise FileExistsError("Another process owns job: " + str(out))
            out.parent.mkdir(parents=True, exist_ok=True)
            print("ASSIST START", record["name"], flush=True)
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       XLA_PYTHON_CLIENT_PREALLOCATE="false",
                       XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops=true")
            with (out.parent / (out.name + ".log")).open("x") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((out / "summary.json").read_text())
        assert summary["status"] == "completed" and summary["job_sha256"] == record["sha256"]
        with (out / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        return dict(name=record["name"], command=command, summary=summary, final_metrics=rows[-1],
                    epoch_time=time.time(), protocol_sha256=digest(root / "protocol.json"))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, item) for item in selected]
        try:
            for future in as_completed(futures):
                result = future.result()
                with (root / "assist_execution.jsonl").open("a") as stream:
                    stream.write(json.dumps(result)+"\n")
                print("ASSIST COMPLETE", json.dumps(result), flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    result = json.loads(manifest.read_text())
    result.update(completed=True, completed_epoch=time.time())
    manifest.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
