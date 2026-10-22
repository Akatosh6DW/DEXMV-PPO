"""Execute an explicit, resumable exploratory matrix, recording every result."""

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


def run_job(command, directory, journal):
    if (directory / "summary.json").exists():
        summary = json.loads((directory / "summary.json").read_text())
        if summary.get("status") != "completed":
            raise ValueError("Incomplete existing result")
    else:
        if directory.exists():
            raise FileExistsError("Partial job needs inspection: " + str(directory))
        directory.parent.mkdir(parents=True, exist_ok=True)
        log = directory.parent / (directory.name + ".log")
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops=true")
        print("START", json.dumps(command), flush=True)
        with log.open("w") as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((directory / "summary.json").read_text())
    with (directory / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    result = dict(run=str(directory), command=command, summary=summary,
                  final_metrics=rows[-1], epoch_time=time.time())
    with journal.open("a") as stream:
        stream.write(json.dumps(result)+"\n")
    print("COMPLETE", json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    for path, expected in protocol.get("source_hashes", {}).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen study source changed: " + path)
    output = ROOT / protocol["output"]
    output.mkdir(parents=True, exist_ok=True)
    journal = output / "execution.jsonl"
    jobs = []
    for job in protocol["jobs"]:
        directory = output / job["name"]
        command = [sys.executable, str(ROOT / "scripts/train_orca_acquisition.py"), "--output", str(directory)]
        for key, value in job["args"].items():
            flag = "--" + key.replace("_", "-")
            if value is True:
                command.append(flag)
            elif value is not False and value is not None:
                command.extend([flag, str(value)])
        jobs.append(run_job(command, directory, journal))
    (output / "matrix_complete.json").write_text(json.dumps(dict(completed=True, jobs=jobs), indent=2)+"\n")


if __name__ == "__main__":
    main()
