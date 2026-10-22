"""Execute one job of the frozen, post-selection portfolio confirmation."""

import argparse
import fcntl
import json
import os
import subprocess
import sys

from confirm_s09_late_experts import ROOT, BASE
from retarget_s09_reference import digest

OUT = BASE/"confirmation_portfolio_20261019"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, choices=("bc", "expert", "portfolio"))
    args = parser.parse_args()
    plan = json.loads((OUT/"plan.json").read_text())
    assert (plan["eval_start"],plan["eval_count"]) == (180000,256)
    assert digest(OUT/"selection.json") == plan["selection_sha256"]
    for path,value in plan["execution_sources"].items():
        assert digest(ROOT/path) == value, path
    for name,key in (("model.xml","model_sha256"),("reference.npz","reference_sha256")):
        assert digest(ROOT/plan["input"]/name) == plan[key]
    job = next(j for j in plan["jobs"] if j["name"] == args.name)
    assert digest(ROOT/job["checkpoint"]) == job["checkpoint_sha256"]
    logs = OUT/"execution"
    logs.mkdir(exist_ok=True)
    lock = (logs/f"{args.name}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_ENABLE_COMPILATION_CACHE="false")
    output = ROOT/job["output"]
    if output.exists() and not (output/"summary.json").exists():
        raise RuntimeError("Partial output exists; inspect it instead of overwriting")
    if not output.exists():
        command = [sys.executable, str(ROOT/job["entry"]), "--input", plan["input"],
            "--output", str(output), "--method", "guided", "--steps", "0",
            "--checkpoint", job["checkpoint"], "--seed", str(job["training_seed"]),
            "--eval-start", str(plan["eval_start"]), "--eval-count", str(plan["eval_count"])]
        print(json.dumps(dict(status="start", name=args.name, command=command)),flush=True)
        with (logs/f"{args.name}.log").open("x") as stream:
            subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    summary = json.loads((output/"summary.json").read_text())
    assert summary["status"] == "completed" and summary["final_step"] == 0
    audit = output/"step_000000000/native_replay_audit.json"
    if not audit.exists():
        with (logs/f"{args.name}_audit.log").open("x") as stream:
            subprocess.run([sys.executable,str(ROOT/job["auditor"]),"--run",str(output)],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    assert json.loads(audit.read_text())["status"] == "verified"
    print(json.dumps(dict(status="completed_and_audited",name=args.name,
        success30=summary["success30"],plan_sha256=digest(OUT/"plan.json"))),flush=True)


if __name__ == "__main__":
    main()
