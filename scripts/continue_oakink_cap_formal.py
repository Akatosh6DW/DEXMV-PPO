"""Wait for the current cap runner, then complete independent tests and artifacts."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from continue_orca_formal import process_identity
from oakink_cap_protocol import ROOT, check_protocol, recipes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--watch-pid", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    suite, output = args.suite.resolve(), args.output.resolve()
    check_protocol(suite / "protocol.json")
    identity = process_identity(args.watch_pid)
    if identity is None or "scripts/run_oakink_cap_formal.py" not in identity["command"]:
        raise ValueError("A live cap training runner is required")
    output.mkdir(parents=True, exist_ok=False)
    (output / "coordinator_source.py").write_bytes(Path(__file__).read_bytes())
    (output / "watched_process.json").write_text(json.dumps(dict(pid=args.watch_pid, **identity), indent=2))

    def event(kind, **values):
        row = dict(event=kind, unix_time=time.time(), **values)
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def execute(label, command):
        with (output / (label + ".log")).open("x") as log:
            env = dict(os.environ, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            process = subprocess.Popen([sys.executable, *command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            event("started", label=label, pid=process.pid, command=command)
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    event("verified_wait", label=label, pid=process.pid)
        event("exited", label=label, code=code)
        if code:
            raise RuntimeError("Cap stage failed, preserved for inspection: " + label)

    with (output / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while process_identity(args.watch_pid) == identity:
            event("verified_wait", label="cap_training", pid=args.watch_pid)
            time.sleep(30)
        check_protocol(suite / "protocol.json")
        complete = json.loads((suite / "complete.json").read_text())
        assert complete["training_jobs"] == 20
        for recipe in recipes():
            assert json.loads((suite / recipe["label"] / "formal_audit.json").read_text())["passed"]
        evaluation = output / "evaluations"
        execute("evaluations", ["scripts/run_oakink_cap_evaluations.py", "--suite", str(suite), "--output", str(evaluation)])
        execute("statistics", ["scripts/analyze_oakink_cap.py", "--suite", str(suite), "--evaluations", str(evaluation),
                               "--output", str(suite / "report")])
        execute("videos", ["scripts/render_oakink_cap_formal.py", "--evaluations", str(evaluation),
                           "--output", str(output / "videos")])
        (output / "complete.json").write_text(json.dumps(dict(cap_workflow_complete=True, entire_goal_complete=False,
                                        note="Whole-project final audit and Chinese synthesis remain separate"), indent=2))


if __name__ == "__main__":
    main()
