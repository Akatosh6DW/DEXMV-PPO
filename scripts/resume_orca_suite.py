"""Resume a stopped S02 matrix, preserving interrupted attempts separately."""

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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    for entry in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = entry.read_bytes().split(b"\0")
        except (OSError, ProcessLookupError):
            continue
        if len(command) > 1 and Path(os.fsdecode(command[1])).name in (
                "run_orca_suite_job.py", "run_orca_suite.py", "assist_orca_suite.py", "finish_orca_suite.py"):
            raise RuntimeError("An existing suite process is still live; observe it instead of restarting")
    for path, expected in protocol["source_hashes"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    archive = root / "interrupted" / str(time.time_ns())
    archived = []
    for record in protocol["jobs"]:
        path = root / record["name"]
        if not path.exists() or (path / "summary.json").exists():
            continue
        target = archive / record["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        metrics = []
        if (path / "metrics.csv").exists():
            with (path / "metrics.csv").open() as stream:
                metrics = list(csv.DictReader(stream))
        path.rename(target)
        log = path.with_suffix(".log")
        if log.exists():
            log.rename(target.with_suffix(".log"))
        archived.append(dict(job=record["name"], archive=str(target.relative_to(root)),
                             recorded_metrics=metrics, job_sha256=record["sha256"],
                             reason="Prior session interrupted; no worker remains; no completion summary"))
    if archived:
        (archive / "manifest.json").write_text(json.dumps(dict(
            epoch_time=time.time(), attempts=archived,
            restart_policy="Original initialization and fixed budget; no partial optimizer state available; do not count interrupted attempts as independent completed trials"), indent=2)+"\n")
    print(json.dumps(dict(archived=archived)), flush=True)

    def run(script, *flags):
        command = [sys.executable, str(ROOT / "scripts" / script), *map(str, flags)]
        print("RESUME COMMAND", json.dumps(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    run("assist_orca_suite.py", "--root", root, "--phase", "train", "--min-seed", "1001", "--workers", "3")
    run("assist_orca_suite.py", "--root", root, "--phase", "eval", "--benchmarks", *protocol["benchmark_names"], "--workers", "4")
    run("run_orca_suite.py", "--protocol", root / "protocol.json")
    run("report_orca_suite.py", "--root", root)
    run("plot_orca_suite.py", "--root", root)
    run("render_orca_suite.py", "--root", root)
    run("audit_orca_suite.py", "--root", root, "--require-videos")
    print("MATRIX AND ARTIFACTS COMPLETE; interpretation still needs review", flush=True)


if __name__ == "__main__":
    main()
