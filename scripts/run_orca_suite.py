"""Sequential resumable runner with exact job and source identity checks."""

import argparse
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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "train", "eval"), default="all")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    for path, expected in protocol["source_hashes"].items():
        if digest(ROOT / path) != expected:
            raise ValueError("Frozen input changed: " + path)
    root = ROOT / protocol["output"]
    journal = root / "execution.jsonl"
    selected = [job for job in protocol["jobs"]
                if args.phase == "all" or job["name"].startswith(args.phase + "/")]
    for index, record in enumerate(selected):
        path = ROOT / record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError("Job changed: " + str(path))
        job = json.loads(path.read_text())
        out = ROOT / job["output"]
        command = [sys.executable, str(ROOT / "scripts/run_orca_suite_job.py"), "--job", str(path)]
        summary_path = out / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary.get("status") != "completed" or summary.get("job_sha256") != record["sha256"]:
                raise ValueError("Existing job identity/completion mismatch")
        else:
            if out.exists():
                raise FileExistsError("Inspect partial job before retry: " + str(out))
            out.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       XLA_PYTHON_CLIENT_PREALLOCATE="false",
                       XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops=true")
            print("START", index+1, len(selected), record["name"], flush=True)
            with (out.parent / (out.name + ".log")).open("w") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            summary = json.loads(summary_path.read_text())
        with (out / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        result = dict(name=record["name"], command=command, summary=summary, final_metrics=rows[-1],
                      epoch_time=time.time(), protocol_sha256=digest(args.protocol))
        with journal.open("a") as stream:
            stream.write(json.dumps(result)+"\n")
        print("COMPLETE", json.dumps(result), flush=True)
    (root / f"{args.phase}_complete.json").write_text(json.dumps(dict(
        completed=True, jobs=[job["name"] for job in selected], protocol_sha256=digest(args.protocol)), indent=2))


if __name__ == "__main__":
    main()
