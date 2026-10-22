"""Drain a known foreground dispatcher and continue this one suite as a service."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from finish_orca_suite import process, suite_processes

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--parent-pid", type=int, required=True)
    args = p.parse_args()
    root = args.root.resolve()

    def record(stage, **fields):
        row = dict(stage=stage, epoch_time=time.time(), **fields)
        with (root / "service.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+"\n")
        print(json.dumps(row), flush=True)

    previous = None
    while True:
        live = suite_processes()
        ids = [r["pid"] for r in live]
        if previous != ids:
            record("waiting_for_existing_workers", processes=live)
            previous = ids
        if not live:
            break
        time.sleep(10)
    parent = process(args.parent_pid)
    if parent and parent["state"] != "Z":
        if len(parent["args"]) < 2 or Path(parent["args"][1]).name != "resume_orca_suite.py":
            raise RuntimeError("Parent identity mismatch; refusing to signal")
        assert (Path("/proc") / str(args.parent_pid) / "cwd").resolve() == ROOT
        os.kill(args.parent_pid, signal.SIGTERM)
        os.kill(args.parent_pid, signal.SIGCONT)
        for _ in range(100):
            row = process(args.parent_pid)
            if row is None or row["state"] == "Z":
                break
            time.sleep(.1)
        else:
            raise RuntimeError("Old dispatcher still live")
    record("service_owns_dispatch", interrupted_running_jobs=False)
    subprocess.run([sys.executable, str(ROOT / "scripts/resume_orca_suite.py"),
                    "--root", str(root)], cwd=ROOT, check=True)
    record("artifacts_complete", final_interpretation_review_required=True)


if __name__ == "__main__":
    main()
