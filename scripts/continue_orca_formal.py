"""Continue frozen training phases and held-out testing after an existing runner."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from orca_formal_protocol import recipes
from run_orca_formal import verify_frozen
from prepare_hic_orca_object import digest

ROOT = Path(__file__).resolve().parents[1]


def process_identity(pid):
    root = Path("/proc") / str(pid)
    try:
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return dict(start=fields[19], command=(root / "cmdline").read_bytes().replace(b"\0", b" ").decode())
    except FileNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--watch-pid", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    suite, output = args.suite.resolve(), args.output.resolve()
    protocol = json.loads((suite / "protocol.json").read_text())
    verify_frozen(protocol)
    identity = process_identity(args.watch_pid)
    if identity is None or "scripts/run_orca_formal.py" not in identity["command"] or "--phase main" not in identity["command"]:
        raise ValueError("Expected a live main-training runner; do not guess from stale files")
    output.mkdir(parents=True, exist_ok=False)
    (output / "coordinator_source.py").write_bytes(Path(__file__).read_bytes())
    (output / "watched_process.json").write_text(json.dumps(dict(pid=args.watch_pid, **identity), indent=2))
    processes = {}

    def event(kind, **values):
        row = dict(event=kind, unix_time=time.time(), **values)
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def start(label, command, cpu=False):
        log = (output / (label + ".log")).open("x")
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", XLA_PYTHON_CLIENT_PREALLOCATE="false")
        if cpu:
            env["JAX_PLATFORMS"] = "cpu"
        else:
            env.pop("JAX_PLATFORMS", None)
        process = subprocess.Popen([sys.executable, *command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        processes[label] = process
        event("started", label=label, pid=process.pid, command=command)
        return process

    def wait(label):
        process = processes[label]
        while True:
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                event("verified_wait", label=label, pid=process.pid)
        event("exited", label=label, code=code)
        if code:
            raise RuntimeError("Stage failed; preserve outputs and inspect: " + label)

    def check_phase(phase):
        verify_frozen(protocol)
        for recipe in recipes():
            if recipe["phase"] != phase:
                continue
            run = suite / recipe["label"]
            checked = json.loads((run / "formal_audit.json").read_text())
            result = json.loads((run / "formal_result.json").read_text())
            if not checked["passed"]:
                raise ValueError("Stage failed audit: " + recipe["label"])
            if result["penetration_max_mm"] > 5 or result["forbidden_fraction"] > .01:
                review = json.loads((run / "contact_review.json").read_text())
                if (not review["review_complete"]
                        or review["disposition"] != "retain_scored_negative_episodes_continue_unchanged_protocol"
                        or review["result_sha256"] != digest(run / "formal_result.json")
                        or review["inspection_sha256"] != digest(run / "contact_inspection.json")
                        or not review["no_score_or_checkpoint_replacement"]
                        or not review["no_environment_reward_or_training_change"]):
                    raise ValueError("Stage still needs contact inspection: " + recipe["label"])

    with (output / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while process_identity(args.watch_pid) == identity:
            event("verified_wait", label="existing_main_runner", pid=args.watch_pid)
            time.sleep(30)
        check_phase("main")
        event("main_training_verified", jobs=40)
        eval_roots = []
        for shard in range(2):
            destination = output / f"main_evaluations_{shard}"
            eval_roots.append(destination)
            start(f"evaluate_main_{shard}", ["scripts/run_orca_formal_evaluations.py", "--suite", str(suite),
                  "--output", str(destination), "--phase", "main", "--shard-count", "2", "--shard-index", str(shard)], cpu=True)
        for phase in ("reference", "exploration"):
            label = "train_" + phase
            start(label, ["scripts/run_orca_formal.py", "--output", str(suite), "--resume", "--phase", phase])
            wait(label)
            check_phase(phase)
        for shard in range(2):
            wait(f"evaluate_main_{shard}")
        for phase in ("reference", "exploration"):
            destination = output / (phase + "_evaluations")
            eval_roots.append(destination)
            label = "evaluate_" + phase
            start(label, ["scripts/run_orca_formal_evaluations.py", "--suite", str(suite), "--output", str(destination),
                          "--phase", phase], cpu=True)
            wait(label)
        start("statistics", ["scripts/analyze_orca_formal.py", "--suite", str(suite), "--evaluations",
                             *map(str, eval_roots), "--output", str(suite / "report")], cpu=True)
        wait("statistics")
        start("videos", ["scripts/render_orca_formal.py", "--evaluations", *map(str, eval_roots[:2]),
                         "--output", str(output / "palm_videos")], cpu=True)
        wait("videos")
        (output / "complete.json").write_text(json.dumps(dict(cube_workflow_complete=True,
                entire_goal_complete=False, note="Cap study and final whole-goal audit remain separate"), indent=2))


if __name__ == "__main__":
    main()
