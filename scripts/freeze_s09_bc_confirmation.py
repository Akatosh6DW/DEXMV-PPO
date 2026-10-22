"""Freeze three completed repair runs and their BC origins before test rollout."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from retarget_s09_reference import digest
from run_s09_study import ROOT, check_protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs=3, type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    check_protocol()
    base = ROOT / "artifacts/orca_benchmark_suite/S09"
    final, bc, old = [], [], []
    def job(name, group, seed, checkpoint, conditioned):
        checkpoint = Path(checkpoint).resolve() if checkpoint else None
        return dict(name=name, group=group, training_seed=seed, conditioned=conditioned,
            output=str((a.output.resolve()/"eval"/name).relative_to(ROOT)),
            method="reference" if checkpoint is None else "guided",
            entry="scripts/run_s09_bc_conditioned.py" if conditioned else "scripts/run_s09_job.py",
            checkpoint=str(checkpoint.relative_to(ROOT)) if checkpoint else None,
            checkpoint_sha256=digest(checkpoint) if checkpoint else None)
    for run in a.runs:
        cfg = json.loads((run/"run_config.json").read_text())
        summary = json.loads((run/"summary.json").read_text())
        assert summary["status"] == "completed" and summary["final_step"] == 163840
        assert cfg["env_config"]["observation_variant"] == "initial_pose_mode_onehot_95_101"
        assert cfg["learning_rate"] == .00003 and cfg["approach_weight"] == 2 and cfg["coverage_weight"] == 2
        seed = cfg["seed"]
        final.append(job(f"new_bc_ppo_s{seed}", "new_bc_ppo", seed, run/"step_000163840/params.pkl", True))
        bc.append(job(f"new_bc_s{seed}", "new_bc", seed, cfg["checkpoint"], True))
        parent = Path(cfg["checkpoint"]).parent.parent
        assert (parent/"phase_curriculum.json").exists()
        bc_cfg = json.loads((parent/"run_config.json").read_text())
        assert bc_cfg["steps"] == 0 and bc_cfg["bc_updates"] == 40000
    assert [r["training_seed"] for r in final] == [13101,13102,13103]
    for seed in (9101,9102,9103):
        old.append(job(f"old_bc_ppo_s{seed}", "old_bc_ppo", seed,
                       base/f"train/guided_s{seed}/step_000163840/params.pkl", False))
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        selection="Final step163840 for all three predetermined training seeds; no test-based selection",
        input=str((base/"inputs").relative_to(ROOT)),
        eval_start=120000, eval_count=128,
        development_range=[110000,110031],
        model_sha256=digest(base/"inputs/model.xml"), reference_sha256=digest(base/"inputs/reference.npz"),
        dataset_sha256=digest(base/"demonstrations/trajectories.npz"),
        criterion="R0 strict released continuous30s, final valid, all failures retained",
        limitations="Same online budget per final PPO run, not same BC/tuning cost or observation interface as old R0",
        execution_sources={str(f.relative_to(ROOT)):digest(f) for f in (
            ROOT/"scripts/run_s09_bc_confirmation.py", ROOT/"scripts/run_s09_bc_conditioned.py",
            ROOT/"scripts/run_s09_bc_repair.py", ROOT/"scripts/audit_s09_bc_repair.py",
            ROOT/"scripts/analyze_s09_bc_confirmation.py")},
        jobs=old+bc+final+[job("reference", "reference", 0, None, False)])
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/"plan.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(path=str(a.output/"plan.json"), sha256=digest(a.output/"plan.json"), jobs=len(manifest["jobs"]))))


if __name__ == "__main__":
    main()
