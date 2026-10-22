"""Run the frozen evaluation commands, with at most three independent workers.

This scheduler does not change the frozen simulator, policies or test protocol.
It refuses incomplete output directories and never overwrites an experiment.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from run_s09_study import OUT, ROOT, check_protocol, jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    logs = OUT / "execution"
    logs.mkdir(exist_ok=True)
    lock = (logs / "evaluation_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = check_protocol()
    work = list(jobs("eval", protocol))
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        PYTHONPATH=str(ROOT / "Aero-Hand/sim_rl/mujoco_playground"))
    guard = threading.Lock()
    stop = threading.Event()

    def event(**row):
        row["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with guard:
            with (logs / "evaluation_events.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

    def execute(command, log, output, phase):
        original = log
        retry = 0
        while log.exists():
            retry += 1
            log = original.with_name(original.stem + f"_retry{retry:02d}" + original.suffix)
        with log.open("x") as handle:
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                stdout=handle, stderr=subprocess.STDOUT)
            event(status="start", phase=phase, pid=process.pid,
                output=str(output), command=command, log=str(log))
            code = process.wait()
        if code:
            raise RuntimeError(f"{phase} failed ({code}): {log}")

    def run(item):
        output, command, expected = item
        if stop.is_set():
            return
        try:
            check_protocol()
            name = "_".join(output.relative_to(OUT).parts)
            summary_file = output / "summary.json"
            if output.exists() and not summary_file.exists():
                raise RuntimeError(f"Inspect partial output, never overwrite: {output}")
            if not output.exists():
                if shutil.disk_usage(ROOT).free < 700 * 1024**2:
                    raise RuntimeError("Free disk below 700MiB reserve")
                execute(command, logs / (name + ".log"), output, "evaluation")
            summary = json.loads(summary_file.read_text())
            assert summary["status"] == "completed" and summary["final_step"] == expected
            folder = output / "step_000000000"
            audit_file = folder / "native_replay_audit.json"
            if not audit_file.exists():
                execute([sys.executable, str(ROOT / "scripts/audit_s09_replay.py"),
                    "--run", str(output)], logs / (name + "_audit.log"), output, "preview_audit")
            audit = json.loads(audit_file.read_text())
            assert audit["status"] == "verified"
            event(status="done", output=str(output), success30=summary["success30"],
                audited_previews=len(audit["episodes"]))
        except Exception as error:
            stop.set()
            event(status="error", output=str(output), error=str(error))
            raise

    # Finish already-running subprocesses on error; do not abandon live jobs.
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run, item) for item in work]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append(str(error))
    if errors:
        raise RuntimeError("; ".join(errors))
    check_protocol()
    event(status="terminal", jobs=len(work), workers=args.workers)


if __name__ == "__main__":
    main()
