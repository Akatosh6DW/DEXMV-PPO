"""Execute the frozen new-video matrix; refuse changed or partial job inputs."""

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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/orca_cylinder_raw_video_20261009"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "eval"), required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    protocol_path = OUT / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol_hash = digest(protocol_path)
    dataset = OUT / "demonstrations/trajectories.npz"
    manifest = OUT / "demonstrations/dataset_manifest.json"
    dataset_hash, manifest_hash = digest(dataset), digest(manifest)
    if json.loads(manifest.read_text())["dataset_sha256"] != dataset_hash:
        raise ValueError("Dataset content changed")
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)

    def verify():
        if digest(protocol_path) != protocol_hash or digest(dataset) != dataset_hash or digest(manifest) != manifest_hash:
            raise ValueError("Study inputs changed during execution")
        for path, expected in protocol["frozen_sha256"].items():
            if digest(ROOT/path) != expected:
                raise ValueError("Frozen source/config/data changed: " + path)

    def run(relative):
        verify()
        path = ROOT/relative
        job = json.loads(path.read_text())
        output = ROOT/job["output"]
        summary = output/"summary.json"
        if summary.exists():
            result = json.loads(summary.read_text())
            if result["job_sha256"] != digest(path) or result["actual_steps"] != job["steps"]:
                raise ValueError("Stale completion")
            return dict(job=relative, already_completed=True, summary=result)
        if output.exists():
            raise FileExistsError("Inspect partial output instead of overwriting: " + str(output))
        if shutil.disk_usage(OUT).free < 2*1024**3:
            raise RuntimeError("Less than 2 GiB disk free")
        log = logs/(path.stem+".log")
        print("START", relative, flush=True)
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops=true")
        with log.open("x") as stream:
            result = subprocess.run([sys.executable, str(ROOT/"scripts/run_orca_raw_can_job.py"),
                                     "--job", str(path)], cwd=ROOT, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode or not summary.exists():
            raise RuntimeError("Job failed: " + str(log))
        verify()
        record = dict(job=relative, summary=json.loads(summary.read_text()))
        print("DONE", json.dumps(record), flush=True)
        return record

    records = []
    with (OUT/"controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run, path) for path in protocol[args.phase+"_jobs"]]
            try:
                for future in as_completed(futures):
                    record = future.result()
                    records.append(record)
                    with (OUT/"execution.jsonl").open("a") as stream:
                        stream.write(json.dumps(record)+"\n")
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        if len(records) != len(protocol[args.phase+"_jobs"]):
            raise RuntimeError("Incomplete matrix")
        result = dict(complete=True, protocol_sha256=protocol_hash, records=records,
                      dataset_sha256=dataset_hash, dataset_manifest_sha256=manifest_hash)
        (OUT/(args.phase+"_completion.json")).write_text(json.dumps(result, indent=2)+"\n")
        print("PHASE_COMPLETE", args.phase, len(records), flush=True)


if __name__ == "__main__":
    main()
