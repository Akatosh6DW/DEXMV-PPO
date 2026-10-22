"""Run frozen quality evaluations, then bind previews to the actual policy."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import run_s09_bc_confirmation as confirmation
from run_s09_study import ROOT
from freeze_s09_ppo_confirmation import HELDOUT_START, HELDOUT_COUNT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--shards",type=int,default=3)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    if (plan["eval_start"],plan["eval_count"])!=(HELDOUT_START,HELDOUT_COUNT):
        parser.error("Use the frozen new heldout set")
    confirmation.main()
    env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",JAX_ENABLE_COMPILATION_CACHE="false")
    for job in plan["jobs"][args.shard::args.shards]:
        run=ROOT/job["output"]
        quality=json.loads((run/"quality_config.json").read_text())
        if not quality["primary_success_unchanged"]:
            raise ValueError("Changed primary criterion")
        failure=json.loads((run/"failure_config.json").read_text())
        if not failure["original_success_unchanged"] or not failure["gate_counts_overlap"]:
            raise ValueError("Missing unchanged-label, nonexclusive failure diagnostics")
        audit=run/"step_000000000/improvement_audit.json"
        if not audit.exists():
            log=args.plan.parent/"execution"/(job["name"]+"_binding.log")
            with log.open("x") as stream:
                subprocess.run([sys.executable,str(ROOT/"scripts/audit_s09_ppo_improvement.py"),
                    "--run",str(run)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        if json.loads(audit.read_text())["status"]!="verified":
            raise ValueError("Policy/reference binding failed")
        print(json.dumps(dict(status="quality_and_binding_verified",name=job["name"])),flush=True)


if __name__ == "__main__":
    main()
