"""Precommit endpoint paths for three origins before reading new test results."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from analyze_s09_late_experts import BASE, ROOT, select_runs
from retarget_s09_reference import digest

OUT = BASE / "confirmation_experts_multisource_20261021"
START, COUNT = 190000, 256
SOURCES = ((14121, "I26_phase_s13101"), (14122, "I27_phase_s13102"), (14123, "I28_phase_s13103"))


def read(path):
    return json.loads(path.read_text())


def prepare():
    assert not OUT.exists()
    conflicts, scanned = [], 0
    for path in (ROOT / "artifacts/orca_benchmark_suite/S09").rglob("*.json"):
        if path.name not in ("run_config.json", "plan.json", "protocol.json"):
            continue
        record = read(path)
        start, count = record.get("eval_start"), record.get("eval_count")
        if isinstance(start, int) and isinstance(count, int):
            scanned += 1
            if start < START + COUNT and start + count > START:
                conflicts.append(str(path))
    if conflicts:
        raise ValueError(f"Evaluation interval already used/reserved: {conflicts}")
    jobs, configs, execution = [], [], set()
    for seed, origin in SOURCES:
        run, control, _, _ = select_runs(long=True) if seed == 14122 else select_runs(long_replica=seed)
        cfg = read(run / "run_config.json")
        old = read(control / "run_config.json")
        for key in ("model_sha256", "reference_sha256", "env_config", "steps", "seed"):
            assert cfg[key] == old[key], key
        assert cfg["steps"] == 655360 and cfg["seed"] == seed
        conversion = read(Path(cfg["checkpoint"]).parent / "conversion.json")
        stage = Path(conversion["source_checkpoint"])
        assert digest(stage) == old["checkpoint_sha256"] == conversion["source_checkpoint_sha256"]
        bc = BASE.parent / origin / "step_000000000/params.pkl"
        previous = Path(read(stage.parent.parent / "run_config.json")["checkpoint"])
        assert Path(read(previous.parent.parent / "run_config.json")["checkpoint"]).resolve() == bc.resolve()
        assert read(bc.parent.parent / "run_config.json")["steps"] == 0
        configs.append(cfg)
        execution.update(ROOT / "scripts" / name for name in cfg["sources"])
        for group, checkpoint, budget, multiplier in (
            ("bc", bc, 0, 1), ("stage", stage, 655360, 1),
            ("single_last", control / "step_000655360/params.pkl", 1310720, 1),
            ("expert_last", run / "step_000655360/params.pkl", 1310720, 6),
        ):
            original = read(checkpoint.parent.parent / "run_config.json")
            name = f"{group}_s{seed}"
            jobs.append(dict(name=name, group=group, training_source=seed, training_seed=original["seed"],
                checkpoint=str(checkpoint.relative_to(ROOT)),
                checkpoint_sha256=digest(checkpoint) if checkpoint.exists() else None,
                endpoint_pending=not checkpoint.exists(),
                source_run_config_sha256=digest(checkpoint.parent.parent / "run_config.json"),
                path_online_steps=budget, actor_parameter_multiplier=multiplier,
                output=str((OUT / "eval" / name).relative_to(ROOT)),
                entry="scripts/evaluate_s09_mode_expert_quality.py" if multiplier == 6 else "scripts/evaluate_s09_failure_quality.py",
                auditor="scripts/audit_s09_mode_experts.py" if multiplier == 6 else "scripts/audit_s09_bc_repair.py"))
    for cfg in configs[1:]:
        for key in ("model_sha256", "reference_sha256", "env_config"):
            assert cfg[key] == configs[0][key]
    execution.update(ROOT / "scripts" / name for name in (
        "evaluate_s09_mode_expert_quality.py", "evaluate_s09_failure_quality.py", "evaluate_s09_quality.py",
        "diagnose_s09_failure_modes.py", "audit_s09_mode_experts.py", "audit_s09_bc_repair.py",
        "analyze_s09_late_experts.py", Path(__file__).name))
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        input="artifacts/orca_benchmark_suite/S09/inputs", eval_start=START, eval_count=COUNT,
        model_sha256=configs[0]["model_sha256"], reference_sha256=configs[0]["reference_sha256"],
        env_config=configs[0]["env_config"], jobs=jobs, scanned_existing_records=scanned,
        execution_sources={str(p.relative_to(ROOT)): digest(p) for p in sorted(execution)},
        selection="All three precommitted source-specific true 655360-step endpoints; no development or test reselection",
        pending_binding="Existing checkpoints hashed now; two running endpoints bind once after completed training and audit, before their evaluation",
        primary_contrast="expert_last minus bc, paired by BC ancestry and test state",
        secondary_contrasts=["expert_last minus stage", "expert_last minus single_last"],
        limitations="Three lineages and 256 common initial states, sixfold actor capacity; no hardware guarantee or residual claim",
        heldout_use="Unconditional endpoint confirmation; do not tune training, rewards, source selection or checkpoints on these results")
    OUT.mkdir()
    (OUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(dict(plan=str(OUT / "plan.json"), sha256=digest(OUT / "plan.json"), jobs=len(jobs))), flush=True)


def execute(name):
    plan = read(OUT / "plan.json")
    assert (plan["eval_start"], plan["eval_count"]) == (START, COUNT)
    job = next(j for j in plan["jobs"] if j["name"] == name)
    for path, sha in plan["execution_sources"].items():
        assert digest(ROOT / path) == sha, path
    for filename, key in (("model.xml", "model_sha256"), ("reference.npz", "reference_sha256")):
        assert digest(ROOT / plan["input"] / filename) == plan[key]
    checkpoint = ROOT / job["checkpoint"]
    parent = checkpoint.parent.parent
    assert digest(parent / "run_config.json") == job["source_run_config_sha256"]
    summary = read(parent / "summary.json")
    assert summary["status"] == "completed"
    if job["group"] in ("single_last", "expert_last"):
        assert summary["final_step"] == 655360
    logs = OUT / "execution"
    logs.mkdir(exist_ok=True)
    lock = (logs / f"{name}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoint_hash = digest(checkpoint)
    if job["checkpoint_sha256"] is not None:
        assert checkpoint_hash == job["checkpoint_sha256"]
    if job["group"] == "expert_last":
        audit = read(checkpoint.parent / "native_replay_audit.json")
        assert audit["status"] == "verified" and audit["checkpoint_sha256"] == checkpoint_hash
    binding_path = logs / f"{name}_binding.json"
    binding = dict(plan_sha256=digest(OUT / "plan.json"), checkpoint_sha256=checkpoint_hash,
                   source_run_config_sha256=job["source_run_config_sha256"], checkpoint=job["checkpoint"])
    if binding_path.exists():
        assert read(binding_path) == binding
    else:
        with binding_path.open("x") as stream:
            json.dump(binding, stream, indent=2)
    output = ROOT / job["output"]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_ENABLE_COMPILATION_CACHE="false")
    if output.exists() and not (output / "summary.json").exists():
        raise RuntimeError("Inspect partial output; do not overwrite or restart a possibly live job")
    if not output.exists():
        assert shutil.disk_usage(ROOT).free > 1024**3
        command = [sys.executable, str(ROOT / job["entry"]), "--input", plan["input"],
            "--output", str(output), "--method", "guided", "--steps", "0", "--checkpoint", str(checkpoint),
            "--eval-start", str(START), "--eval-count", str(COUNT), "--seed", str(job["training_seed"]), "--std", ".03"]
        print(json.dumps(dict(status="start", name=name, command=command)), flush=True)
        with (logs / f"{name}.log").open("x") as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    summary = read(output / "summary.json")
    assert summary["status"] == "completed" and summary["final_step"] == 0
    audit_path = output / "step_000000000/native_replay_audit.json"
    if not audit_path.exists():
        with (logs / f"{name}_audit.log").open("x") as stream:
            subprocess.run([sys.executable, str(ROOT / job["auditor"]), "--run", str(output), "--reproduce-evaluation"],
                           cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    audit = read(audit_path)
    assert audit["status"] == "verified"
    assert audit["checkpoint_sha256"] == digest(output / "step_000000000/params.pkl")
    assert digest(checkpoint) == checkpoint_hash
    print(json.dumps(dict(status="completed_and_audited", name=name, plan_sha256=digest(OUT / "plan.json"))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run"))
    parser.add_argument("--names", nargs="+", choices=[f"{g}_s{s}" for s, _ in SOURCES
                        for g in ("bc", "stage", "single_last", "expert_last")])
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare()
    elif args.names:
        for name in args.names:
            execute(name)
    else:
        parser.error("run requires --names")
