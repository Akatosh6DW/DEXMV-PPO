"""Evaluate a frozen repair manifest, without selecting models on test results.

Each shard runs sequentially. Multiple authorized shell sessions may run
different shards; an advisory lock prevents duplicate execution of one shard.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from retarget_s09_reference import digest
from run_s09_study import ROOT, check_protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=3)
    a = p.parse_args()
    if not 0 <= a.shard < a.shards:
        p.error("Invalid shard")
    plan = json.loads(a.plan.read_text())
    check_protocol()
    for path, expected in plan["execution_sources"].items():
        assert digest(ROOT/path) == expected, "Changed frozen evaluation source: " + path
    execution = a.plan.parent / "execution"
    execution.mkdir(exist_ok=True)
    lock = (execution / f"shard_{a.shard}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_ENABLE_COMPILATION_CACHE="false")
    for job in plan["jobs"][a.shard::a.shards]:
        output = ROOT / job["output"]
        if job.get("checkpoint"):
            assert digest(ROOT / job["checkpoint"]) == job["checkpoint_sha256"]
        command = [sys.executable, str(ROOT / job["entry"]), "--input", plan["input"],
                   "--output", str(output), "--method", job["method"], "--steps", "0",
                   "--eval-count", str(plan["eval_count"]), "--eval-start", str(plan["eval_start"]),
                   "--seed", str(job["training_seed"])]
        if job.get("checkpoint"):
            command += ["--checkpoint", job["checkpoint"]]
        if output.exists() and not (output / "summary.json").exists():
            raise RuntimeError(f"Inspect partial output; never overwrite: {output}")
        if not output.exists():
            if shutil.disk_usage(ROOT).free < 700 * 1024**2:
                raise RuntimeError("Keep at least 700MiB disk reserve")
            log = execution / (job["name"] + ".log")
            print(json.dumps(dict(status="start", name=job["name"], command=command)), flush=True)
            with log.open("x") as f:
                subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((output / "summary.json").read_text())
        assert summary["status"] == "completed" and summary["final_step"] == 0
        audit_path = output / "step_000000000/native_replay_audit.json"
        if not audit_path.exists():
            auditor = "audit_s09_bc_repair.py" if job["conditioned"] else "audit_s09_replay.py"
            with (execution / (job["name"] + "_audit.log")).open("x") as f:
                subprocess.run([sys.executable, str(ROOT / "scripts" / auditor), "--run", str(output)],
                               cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        assert json.loads(audit_path.read_text())["status"] == "verified"
        print(json.dumps(dict(status="done", name=job["name"], success30=summary["success30"])), flush=True)
    print(json.dumps(dict(status="shard_completed", shard=a.shard, plan_sha256=digest(a.plan))), flush=True)


if __name__ == "__main__":
    main()
