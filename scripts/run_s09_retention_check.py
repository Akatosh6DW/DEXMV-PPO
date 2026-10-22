"""Execute one fixed retention-check job, then audit its saved previews."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018/confirmation_retention_20261019/plan.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True,
                        choices=("bc", "stage_start", "control_last", "control_selected", "candidate"))
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for path, expected in plan["execution_sources"].items():
        assert sha(ROOT/path) == expected, path
    job = next(job for job in plan["jobs"] if job["name"] == args.name)
    assert sha(ROOT/job["checkpoint"]) == job["checkpoint_sha256"]
    for name in ("model", "reference"):
        path = ROOT/plan["input"]/("model.xml" if name == "model" else "reference.npz")
        assert sha(path) == plan[f"{name}_sha256"]
    output = ROOT/job["output"]
    if output.exists():
        raise ValueError("Do not silently overwrite or resume a partial evaluation")
    command = [sys.executable, "scripts/evaluate_s09_failure_quality.py", "--input", plan["input"],
        "--method", "guided", "--checkpoint", job["checkpoint"], "--seed", str(plan["evaluation_initialization_seed"]),
        "--std", ".03", "--steps", "0", "--eval-count", str(plan["eval_count"]),
        "--eval-start", str(plan["eval_start"]), "--output", job["output"]]
    print(json.dumps(dict(job=args.name, plan_sha256=sha(PLAN), command=command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    # Keep the original audit tolerance. A numerical-layout failure must be
    # diagnosed explicitly, never retried with a looser threshold here.
    subprocess.run([sys.executable, "scripts/audit_s09_bc_repair.py", "--run", job["output"], "--step", "0"],
                   cwd=ROOT, check=True)
    (output/"retention_check_binding.json").write_text(json.dumps(dict(
        plan_sha256=sha(PLAN), checkpoint_sha256=job["checkpoint_sha256"],
        source_sha256=sha(Path(__file__)), completed_with_preview_audit=True), indent=2)+"\n")


if __name__ == "__main__":
    main()
