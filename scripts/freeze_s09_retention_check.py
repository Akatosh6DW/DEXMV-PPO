"""Freeze a single-training-repetition retention check before new initial states.

This is not a replacement for the earlier three-repetition confirmation.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"
OUTPUT = BASE/"confirmation_retention_20261019"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise ValueError("Never overwrite a frozen check")
    for path in (ROOT/"artifacts/orca_benchmark_suite/S09").rglob("run_config.json"):
        cfg = json.loads(path.read_text())
        start, count = cfg.get("eval_start"), cfg.get("eval_count")
        if isinstance(start, int) and isinstance(count, int) and max(start, 160000) < min(start+count, 160256):
            raise ValueError(f"Proposed initial states already used: {path}")
    old_plan_path = BASE/"confirmation_20261019/plan.json"
    old_plan = json.loads(old_plan_path.read_text())
    sources = old_plan["execution_sources"]
    for path, expected in sources.items():
        assert digest(ROOT/path) == expected, path
    candidate = BASE/"E72_bc_stage_retention"
    summary = json.loads((candidate/"summary.json").read_text())
    assert summary["status"] == "completed" and summary["final_step"] == 655360
    audit_path = candidate/"step_000655360/native_replay_audit.json"
    audit = json.loads(audit_path.read_text())
    assert audit["status"] == "verified"
    selections = [
        ("bc", BASE.parent/"I27_phase_s13102/step_000000000/params.pkl", 0),
        ("stage_start", BASE/"E30_bc_fast_critic_continue_s14112/step_000491520/params.pkl", 655360),
        ("control_last", BASE/"E46_bc_budget_continue_s14122/step_000655360/params.pkl", 1310720),
        ("control_selected", BASE/"E46_bc_budget_continue_s14122/step_000327680/params.pkl", 983040),
        ("candidate", candidate/"step_000655360/params.pkl", 1310720),
    ]
    expected = json.loads((candidate/"run_config.json").read_text())
    jobs = []
    for name, checkpoint, budget in selections:
        cfg = json.loads((checkpoint.parent.parent/"run_config.json").read_text())
        for key in ("model_sha256", "reference_sha256", "env_config", "actor_dim", "critic_dim", "method"):
            assert cfg[key] == expected[key], (name, key)
        jobs.append(dict(name=name, checkpoint=str(checkpoint.relative_to(ROOT)),
            checkpoint_sha256=digest(checkpoint), path_interactions=budget,
            output=str((OUTPUT/"eval"/name).relative_to(ROOT))))
    assert jobs[-1]["checkpoint_sha256"] == audit["checkpoint_sha256"]
    selected_path = jobs[3]["checkpoint"]
    assert any(job.get("checkpoint") == selected_path for job in old_plan["jobs"])
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        eval_start=160000, eval_count=256, training_repetitions=1, bc_seed=13102,
        evaluation_initialization_seed=14102,
        input=str(Path(expected["input"]).relative_to(ROOT)),
        model_sha256=expected["model_sha256"], reference_sha256=expected["reference_sha256"],
        execution_sources=sources, prior_confirmation_sha256=digest(old_plan_path),
        selection="E72 actual endpoint, fixed before new test; all five comparators retained",
        selected_control="E46 checkpoint already selected by the previous frozen development rule",
        limitations="One training repetition; uncertainty across initial conditions is not training stability. No new RGB extraction or altered task.",
        primary="unchanged S09 success30; failures retained",
        candidate_training_config_sha256=digest(candidate/"mode_anchor_config.json"),
        candidate_preview_audit_sha256=digest(audit_path),
        freezer_sha256=digest(Path(__file__)), jobs=jobs)
    OUTPUT.mkdir()
    path = OUTPUT/"plan.json"
    path.write_text(json.dumps(plan, indent=2)+"\n")
    print(json.dumps(dict(plan=str(path), sha256=digest(path), jobs=len(jobs))))


if __name__ == "__main__":
    main()
