"""Drain the sequential training runner, then finish evaluation and artifacts.

Only the orchestration parent is stopped, after preserving every worker's output.
No policy, test case, training budget or frozen experiment source is changed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def process(pid):
    base = Path("/proc") / str(pid)
    try:
        args = [os.fsdecode(x) for x in (base / "cmdline").read_bytes().split(b"\0") if x]
        status = dict(line.split(":", 1) for line in (base / "status").read_text().splitlines() if ":" in line)
        return dict(pid=int(pid), args=args, state=status["State"].strip().split()[0])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def suite_processes():
    result = []
    for path in Path("/proc").glob("[0-9]*"):
        row = process(path.name)
        if row and row["state"] != "Z" and len(row["args"]) > 1:
            if Path(row["args"][1]).name in ("run_orca_suite_job.py", "assist_orca_suite.py"):
                result.append(row)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--runner-pid", type=int, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    journal = root / "handoff.jsonl"

    def record(stage, **extra):
        row = dict(stage=stage, epoch_time=time.time(), **extra)
        with journal.open("a") as stream:
            stream.write(json.dumps(row)+"\n")
        print(json.dumps(row), flush=True)

    def checked_parent():
        row = process(args.runner_pid)
        if row is None or row["state"] == "Z":
            return None
        if len(row["args"]) < 2 or Path(row["args"][1]).name != "run_orca_suite.py":
            raise RuntimeError("Refusing to signal a different process")
        if (Path("/proc") / str(args.runner_pid) / "cwd").resolve() != ROOT:
            raise RuntimeError("Runner workspace mismatch")
        return row

    def run(script, *flags):
        command = [sys.executable, str(ROOT / "scripts" / script), *map(str, flags)]
        record("command_start", command=command)
        subprocess.run(command, cwd=ROOT, check=True)
        record("command_complete", command=command)

    record("waiting_for_training", runner_pid=args.runner_pid,
           orchestration_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    previous = None
    while True:
        missing = []
        for job in protocol["jobs"]:
            if not job["name"].startswith("train/"):
                continue
            summary = root / job["name"] / "summary.json"
            if not summary.exists():
                missing.append(job["name"])
                continue
            data = json.loads(summary.read_text())
            assert data["status"] == "completed" and data["job_sha256"] == job["sha256"]
            assert data["final_step"] == protocol["training_steps"]
        if len(missing) != previous:
            record("training_progress", completed=50-len(missing), missing=missing)
            previous = len(missing)
        if not missing:
            break
        if checked_parent() is None:
            raise RuntimeError("Sequential parent ended before all training completed; inspect before retry")
        time.sleep(10)
    stopped = False
    try:
        if checked_parent():
            os.kill(args.runner_pid, signal.SIGSTOP)
            stopped = True
            record("parent_dispatch_stopped", pid=args.runner_pid)
        prior = None
        while True:
            live = suite_processes()
            ids = [r["pid"] for r in live]
            if ids != prior:
                record("draining_existing_workers", processes=live)
                prior = ids
            if not live:
                break
            time.sleep(10)
        if checked_parent():
            os.kill(args.runner_pid, signal.SIGTERM)
            os.kill(args.runner_pid, signal.SIGCONT)
            for _ in range(100):
                if checked_parent() is None:
                    break
                time.sleep(.1)
            else:
                raise RuntimeError("Orchestration parent did not exit")
        stopped = False
        record("parent_handoff_complete", no_worker_interrupted=True)
    finally:
        if stopped and checked_parent():
            os.kill(args.runner_pid, signal.SIGCONT)
            record("parent_resumed_after_handoff_failure")

    run("assist_orca_suite.py", "--root", root, "--phase", "eval", "--benchmarks",
        *protocol["benchmark_names"], "--workers", "4")
    # Canonical sequential verifier now sees only completed jobs and launches none.
    run("run_orca_suite.py", "--protocol", root / "protocol.json")
    run("report_orca_suite.py", "--root", root)
    run("plot_orca_suite.py", "--root", root)
    run("render_orca_suite.py", "--root", root)
    run("audit_orca_suite.py", "--root", root, "--require-videos")
    record("matrix_and_artifacts_complete", interpretation_document_still_requires_review=True)


if __name__ == "__main__":
    main()
