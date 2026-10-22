"""Freeze and execute one-origin expert endpoint confirmation without reselection."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from retarget_s09_reference import digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"
OUT = BASE/"confirmation_experts_20261019"


def freeze():
    read = lambda p: json.loads(p.read_text())
    expert = BASE/"E78_bc_late_experts_long"
    control = BASE/"E46_bc_budget_continue_s14122"
    summary = read(expert/"summary.json")
    assert summary["status"] == "completed" and summary["final_step"] == 655360
    assert read(expert/"step_000655360/native_replay_audit.json")["status"] == "verified"
    cfg, old = read(expert/"run_config.json"), read(control/"run_config.json")
    for key in ("model_sha256", "reference_sha256", "env_config", "steps", "seed"):
        assert cfg[key] == old[key]
    conversion = read(Path(cfg["checkpoint"]).parent/"conversion.json")
    assert conversion["source_checkpoint_sha256"] == old["checkpoint_sha256"]
    source = Path(conversion["source_checkpoint"])
    assert digest(source) == old["checkpoint_sha256"]
    source_run = source.parent.parent
    previous = Path(read(source_run/"run_config.json")["checkpoint"])
    bc = Path(read(previous.parent.parent/"run_config.json")["checkpoint"])
    assert bc.parent.parent.name == "I27_phase_s13102"
    assert read(bc.parent.parent/"run_config.json")["steps"] == 0
    jobs = []
    for name, checkpoint, budget, multiplier in (
        ("bc", bc, 0, 1), ("stage", source, 655360, 1),
        ("single_last", control/"step_000655360/params.pkl", 1310720, 1),
        ("expert_last", expert/"step_000655360/params.pkl", 1310720, 6)):
        original = read(checkpoint.parent.parent/"run_config.json")
        jobs.append(dict(name=name, checkpoint=str(checkpoint.relative_to(ROOT)),
            checkpoint_sha256=digest(checkpoint), training_seed=original["seed"],
            path_online_steps=budget, actor_parameter_multiplier=multiplier,
            output=str((OUT/"eval"/name).relative_to(ROOT)),
            entry="scripts/evaluate_s09_mode_expert_quality.py" if multiplier == 6 else "scripts/evaluate_s09_failure_quality.py",
            auditor="scripts/audit_s09_mode_experts.py" if multiplier == 6 else "scripts/audit_s09_bc_repair.py"))
    sources = {ROOT/"scripts"/name for name in cfg["sources"]}
    sources.update(ROOT/"scripts"/name for name in (
        "evaluate_s09_mode_expert_quality.py", "evaluate_s09_failure_quality.py", "evaluate_s09_quality.py",
        "diagnose_s09_failure_modes.py", "audit_s09_mode_experts.py", "audit_s09_bc_repair.py", Path(__file__).name))
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        input="artifacts/orca_benchmark_suite/S09/inputs", eval_start=170000, eval_count=256,
        model_sha256=cfg["model_sha256"], reference_sha256=cfg["reference_sha256"],
        env_config=cfg["env_config"], jobs=jobs,
        execution_sources={str(p.relative_to(ROOT)):digest(p) for p in sorted(sources)},
        selection="E78 true 655360-step endpoint fixed before heldout execution; no best-checkpoint substitution",
        comparisons=["expert_last minus bc", "expert_last minus stage", "expert_last minus single_last"],
        criterion="Unchanged S09 strict released continuous30s with final valid and original contact gates",
        limitations="One BC/training origin, sixfold actor capacity; not independent training replications or equal search cost. 256 paired states measure initial-state uncertainty only.",
        heldout_use="One fixed evaluation; do not choose checkpoints, rewards, or hyperparameters using these results")
    OUT.mkdir(exist_ok=False)
    (OUT/"plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    print(json.dumps(dict(plan=str(OUT/"plan.json"), sha256=digest(OUT/"plan.json"), jobs=len(jobs))))


def execute(name):
    plan_path = OUT/"plan.json"
    plan = json.loads(plan_path.read_text())
    assert (plan["eval_start"], plan["eval_count"]) == (170000, 256)
    for path, expected in plan["execution_sources"].items():
        assert digest(ROOT/path) == expected, path
    for file, key in (("model.xml", "model_sha256"), ("reference.npz", "reference_sha256")):
        assert digest(ROOT/plan["input"]/file) == plan[key]
    job = next(j for j in plan["jobs"] if j["name"] == name)
    assert digest(ROOT/job["checkpoint"]) == job["checkpoint_sha256"]
    logs = OUT/"execution"
    logs.mkdir(exist_ok=True)
    lock = (logs/f"{name}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_ENABLE_COMPILATION_CACHE="false")
    output = ROOT/job["output"]
    if output.exists() and not (output/"summary.json").exists():
        raise RuntimeError("Inspect partial output, do not overwrite")
    if not output.exists():
        assert shutil.disk_usage(ROOT).free > 1024**3
        command = [sys.executable, str(ROOT/job["entry"]), "--input", plan["input"],
            "--output", str(output), "--method", "guided", "--steps", "0", "--checkpoint", job["checkpoint"],
            "--eval-start", str(plan["eval_start"]), "--eval-count", str(plan["eval_count"]),
            "--seed", str(job["training_seed"])]
        print(json.dumps(dict(status="start", name=name, command=command)), flush=True)
        with (logs/f"{name}.log").open("x") as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    summary = json.loads((output/"summary.json").read_text())
    assert summary["status"] == "completed" and summary["final_step"] == 0
    audit = output/"step_000000000/native_replay_audit.json"
    if not audit.exists():
        with (logs/f"{name}_audit.log").open("x") as stream:
            subprocess.run([sys.executable, str(ROOT/job["auditor"]), "--run", str(output)],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    assert json.loads(audit.read_text())["status"] == "verified"
    print(json.dumps(dict(status="completed_and_audited", name=name, success30=summary["success30"],
                         plan_sha256=digest(plan_path))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("freeze", "run"))
    parser.add_argument("--name", choices=("bc", "stage", "single_last", "expert_last"))
    args = parser.parse_args()
    if args.operation == "freeze":
        freeze()
    elif args.name:
        execute(args.name)
    else:
        parser.error("run requires --name")
