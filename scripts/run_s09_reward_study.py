"""Paired reward-only continuation and fresh exploratory tests, AFTER frozen R0."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
import shutil
import subprocess
import sys
import threading
import time

from retarget_s09_reference import digest
from run_s09_study import METHODS, OUT, ROOT, check_protocol

DEST = OUT / "reward_iteration"
STRATA = {"uniform": (100000, 32), "web": (100200, 16)}
VERSIONS = ("original", "continued_R0", "penalty_R1")


def protocol():
    r0 = check_protocol()
    verified = json.loads((OUT / "report/matrix_verification.json").read_text())
    assert verified["status"] == "matrix_verified" and verified["episodes"] == 2560
    DEST.mkdir(exist_ok=True)
    expected = dict(version="S09-R1-exploratory", parent_protocol_sha256=digest(OUT / "protocol.json"),
        initialization_seed=9101, continuation_seed=10101, extra_steps=40960,
        development_seeds=list(range(92000, 92008)), test_strata=STRATA,
        penalty=dict(weight=5, start_m=.0015, scale_m=.0015, cap=4),
        recipes=r0["recipes"], methods=list(METHODS), versions=VERSIONS,
        sources={name: digest(ROOT / "scripts" / name) for name in
            ("run_s09_reward_job.py", "run_s09_reward_study.py")},
        checkpoints={method: digest(OUT / "train" / f"{method}_s9101/step_000163840/params.pkl")
            for method in METHODS},
        checkpoint_selection="Fixed final extra step40960, not selected by test outcomes",
        limitation="One paired continuation seed; exploratory, not a replacement for 3-seed R0")
    expected = json.loads(json.dumps(expected))
    path = DEST / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == expected, "R1 protocol changed"
    else:
        path.write_text(json.dumps(expected, indent=2) + "\n")
    return expected


def tasks(phase, plan):
    base = [sys.executable, str(ROOT / "scripts/run_s09_job.py"), "--input", str(OUT / "inputs")]
    if phase == "train":
        for version in VERSIONS[1:]:
            for method in METHODS:
                output = DEST / "train" / f"{version}_{method}"
                command = list(base)
                if version == "penalty_R1":
                    command[1] = str(ROOT / "scripts/run_s09_reward_job.py")
                    command += ["--penalty-weight", "5"]
                command += ["--output", str(output), "--method", method, "--seed", "10101",
                    "--checkpoint", str(OUT / "train" / f"{method}_s9101/step_000163840/params.pkl"),
                    "--steps", "40960", "--evals", "3", "--eval-count", "8", "--eval-start", "92000",
                    "--learning-rate", str(plan["recipes"][method]["learning_rate"]), "--std", ".3"]
                yield output, command, 40960
    else:
        for orientation, (start, count) in STRATA.items():
            for method in (*METHODS, "reference"):
                for version in (("original",) if method == "reference" else VERSIONS):
                    name = "reference" if method == "reference" else f"{version}_{method}"
                    output = DEST / "eval" / orientation / name
                    command = base + ["--output", str(output), "--method", method, "--steps", "0",
                        "--orientation", orientation, "--eval-count", str(count), "--eval-start", str(start)]
                    if method != "reference":
                        run = (OUT / "train" / f"{method}_s9101" if version == "original"
                               else DEST / "train" / name)
                        step = 163840 if version == "original" else 40960
                        summary = json.loads((run / "summary.json").read_text())
                        assert summary["status"] == "completed" and summary["final_step"] == step
                        command += ["--checkpoint", str(run / f"step_{step:09d}/params.pkl"),
                                    "--seed", "9101" if version == "original" else "10101"]
                    yield output, command, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "eval"))
    args = parser.parse_args()
    plan = protocol()
    logs = DEST / "execution"
    logs.mkdir(exist_ok=True)
    lock = (logs / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONPATH=str(ROOT / "Aero-Hand/sim_rl/mujoco_playground"))
    stop, guard = threading.Event(), threading.Lock()

    def event(**row):
        row["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with guard:
            with (logs / "events.jsonl").open("a") as file:
                file.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

    def run(item):
        output, command, expected = item
        if stop.is_set():
            return
        try:
            name = "_".join(output.relative_to(DEST).parts)
            summary_file = output / "summary.json"
            if output.exists() and not summary_file.exists():
                raise RuntimeError(f"Inspect partial output: {output}")
            if not output.exists():
                if shutil.disk_usage(ROOT).free < 700 * 1024**2:
                    raise RuntimeError("Free disk below 700MiB reserve")
                check_protocol()
                with (logs / (name + ".log")).open("x") as file:
                    process = subprocess.Popen(command, cwd=ROOT, env=environment,
                        stdout=file, stderr=subprocess.STDOUT)
                    event(status="start", output=str(output), pid=process.pid, command=command)
                    code = process.wait()
                if code:
                    raise RuntimeError(f"Exit {code}: {logs / (name + '.log')}")
            summary = json.loads(summary_file.read_text())
            assert summary["status"] == "completed" and summary["final_step"] == expected
            with (output / "metrics.csv").open() as file:
                metrics = list(csv.DictReader(file))
            assert float(metrics[-1]["success30"]) == summary["success30"]["last"]
            if args.phase == "eval" and not (output / "step_000000000/native_replay_audit.json").exists():
                with (logs / (name + "_audit.log")).open("x") as file:
                    subprocess.run([sys.executable, str(ROOT / "scripts/audit_s09_replay.py"),
                        "--run", str(output)], cwd=ROOT, env=environment, stdout=file,
                        stderr=subprocess.STDOUT, check=True)
            event(status="done", output=str(output), success30=summary["success30"],
                  hold_sec=summary["hold_sec"], last_metrics=metrics[-1],
                  next_action="Next preregistered job; no adaptive recipe change")
        except Exception as error:
            stop.set()
            event(status="error", output=str(output), error=str(error))
            raise

    work = list(tasks(args.phase, plan))
    errors = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        for future in as_completed([executor.submit(run, task) for task in work]):
            try:
                future.result()
            except Exception as error:
                errors.append(str(error))
    if errors:
        raise RuntimeError("; ".join(errors))
    protocol()
    event(status="terminal", phase=args.phase, jobs=len(work))


if __name__ == "__main__":
    main()
