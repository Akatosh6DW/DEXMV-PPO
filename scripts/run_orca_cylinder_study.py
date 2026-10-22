"""Run a frozen cylinder matrix with bounded concurrency and resumable audits."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/orca_cylinder_study"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "eval"), required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    protocol_path = OUT / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    records, lock = [], threading.Lock()
    jobs = protocol[args.phase + "_jobs"]
    (OUT / "logs").mkdir(exist_ok=True)

    def verify_inputs():
        if digest(protocol_path) != protocol_hash:
            raise ValueError("Protocol changed during execution")
        for path, expected in protocol["frozen_inputs"].items():
            if digest(ROOT / path) != expected:
                raise ValueError("Frozen input changed: " + path)

    def run(job_path):
        verify_inputs()
        path = ROOT / job_path
        job = json.loads(path.read_text())
        if digest(path) != protocol["job_sha256"][job_path]:
            raise ValueError("Job changed")
        output = ROOT / job["output"]
        summary = output / "summary.json"
        if summary.exists():
            result = json.loads(summary.read_text())
            if result["job_sha256"] != digest(path) or result["actual_steps"] != job["steps"]:
                raise ValueError("Existing completion does not match job")
            return dict(job=job_path, skipped=True, summary=result)
        if output.exists():
            raise FileExistsError("Inspect incomplete output before retry: " + str(output))
        if shutil.disk_usage(OUT).free < 2 * 1024**3:
            raise RuntimeError("Less than 2 GiB free: stop before creating more experiment data")
        log = OUT / "logs" / (path.stem + ".log")
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops=true")
        print("START", job_path, flush=True)
        with log.open("x") as stream:
            completed = subprocess.run([sys.executable, str(ROOT / "scripts/run_orca_cylinder_job.py"),
                                        "--job", str(path)], cwd=ROOT, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode or not summary.exists():
            raise RuntimeError("Job failed; inspect " + str(log))
        result = json.loads(summary.read_text())
        verify_inputs()
        record = dict(job=job_path, skipped=False, summary=result)
        with lock:
            with (OUT / "execution.jsonl").open("a") as stream:
                stream.write(json.dumps(record)+"\n")
        print("DONE", json.dumps(record), flush=True)
        return record

    protocol_hash = digest(protocol_path)
    with (OUT / "controller.lock").open("a") as controller:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_inputs()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run, path) for path in jobs]
            try:
                for future in as_completed(futures):
                    records.append(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        if len(records) != len(jobs):
            raise RuntimeError("Incomplete phase")
        (OUT / f"{args.phase}_completion.json").write_text(json.dumps(dict(
            complete=True, protocol_sha256=protocol_hash, records=records), indent=2)+"\n")
        print("PHASE_COMPLETE", args.phase, len(records), flush=True)


if __name__ == "__main__":
    main()
