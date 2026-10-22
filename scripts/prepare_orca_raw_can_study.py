"""Freeze a fresh-video study before replay scores or training are inspected."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/orca_cylinder_raw_video_20261009"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        raise FileExistsError("Do not replace a frozen protocol")
    paths = [OUT / name for name in (
        "source/tomato41_raw.mp4", "perception/hand_landmarker.task",
        "perception/tomato41/hand_landmarks.npz", "perception/tomato41/manifest.json",
        "perception/can41_v2/hand_object.npz", "perception/can41_v2/manifest.json",
        "retarget_neutral/model.xml", "retarget_neutral/reference.npz", "retarget_neutral/manifest.json")]
    paths += [ROOT / "scripts" / name for name in (
        "extract_raw_video_hand.py", "extract_raw_can_geometry.py", "retarget_raw_can_orca.py",
        "orca_raw_can_env.py", "orca_raw_can_bc.py", "run_orca_raw_can_job.py", "prepare_orca_raw_can_study.py",
        "orca_cylinder_evaluation.py", "orca_native_physics.py", "orca_suite_env.py",
        "orca_acquisition_env.py", "orca_matched_env.py", "orca_hic53_env.py",
        "orca_refit_env.py", "orca_residual_env.py", "orca_benchmark_env.py",
        "orca_residual_controls.py", "orca_contact_forces.py", "train_orca_matched.py",
        "train_orca_comparison.py", "train_orca_acquisition.py", "run_orca_suite_job.py")]
    prefix = str(OUT.relative_to(ROOT))
    base = dict(model=prefix+"/retarget_neutral/model.xml", reference=prefix+"/retarget_neutral/reference.npz",
                benchmark="B0_nominal", std=.3, learning_rate=3e-4, eval_start=161000,
                eval_count=16, evals=5)
    jobs = OUT / "jobs"
    jobs.mkdir(exist_ok=False)
    def save(name, **values):
        path = jobs / (name+".json")
        path.write_text(json.dumps(dict(base, **values), indent=2)+"\n")
        return str(path.relative_to(ROOT))
    collect = save("collect", method="collect", steps=0, seed=51000,
                   eval_start=151000, eval_count=32, output=prefix+"/demonstrations")
    dataset = prefix+"/demonstrations/trajectories.npz"
    training, evaluations = [], []
    for seed in (51001, 51002, 51003):
        for method in ("ppo", "guided", "residual"):
            name = f"{method}_s{seed}"
            run = prefix+"/train/"+name
            training.append(save("train_"+name, method=method, seed=seed, steps=655360,
                                  output=run, dataset=dataset))
            for budget in (163840, 655360):
                for condition in ("B0_nominal", "B1_position", "B5_push"):
                    label = f"{name}_{budget}_{condition}"
                    evaluations.append(save("eval_"+label, method=method, seed=seed, steps=0,
                                            budget=budget, benchmark=condition, eval_start=210000,
                                            eval_count=64, checkpoint=run+f"/step_{budget:09d}/params.pkl",
                                            output=prefix+"/eval/"+label))
            if method == "guided":
                for condition in ("B0_nominal", "B1_position", "B5_push"):
                    label = f"bc_s{seed}_{condition}"
                    evaluations.append(save("eval_"+label, method="bc", seed=seed, steps=0,
                                            budget=0, benchmark=condition, eval_start=210000, eval_count=64,
                                            checkpoint=run+"/bc_params.pkl", output=prefix+"/eval/"+label))
    for condition in ("B0_nominal", "B1_position", "B5_push"):
        evaluations.append(save("eval_zero_"+condition, method="zero", seed=51000, steps=0,
                                budget=0, benchmark=condition, eval_start=210000, eval_count=64,
                                output=prefix+"/eval/zero_"+condition))
    paths += list(jobs.glob("*.json"))
    protocol = dict(status="v2_refrozen_after_BC_interface_fix_before_fresh_training", collection_job=collect,
                    train_jobs=training, eval_jobs=evaluations,
                    frozen_sha256={str(path.relative_to(ROOT)):digest(path) for path in paths},
                    trained_seeds=[51001, 51002, 51003], online_budgets=[163840, 655360],
                    episode_seconds=40, hold_seconds=30, residual_bound_rad=.08,
                    common_learning_rate=3e-4, common_initial_std=.3, bc_updates=4000,
                    dataset_rule="Keep all 32 fresh replay trajectories including failures; active samples only for BC",
                    no_reward_search=True, no_resume_old_weights=True, no_old_pose_inputs=True,
                    amendment="v1 loader rejected missing late/holding samples; v2 uses uniform active samples. Same model/reference/reward/budgets; interrupted v1 weights not reused.",
                    results_may_all_fail=True, selection="Final checkpoints at fixed budgets, never best-performing selection",
                    primary_metric="30s uninterrupted valid hold, valid finish, no post-release forbidden contact or >3mm penetration",
                    limitations=["Only one raw source video, not generalization across videos",
                                 "Monocular approximate hand/object reconstruction; large retarget errors already observed",
                                 "No reward-driven reference repair allowed",
                                 "Same LR/std across methods is one fixed recipe, not optimized algorithm comparison",
                                 "Holding benchmark with fixed wrist, not full human transport task",
                                 "Target matches source collision geometry dimensions, mass assumed 35g"])
    protocol_path.write_text(json.dumps(protocol, indent=2)+"\n")
    print(json.dumps(dict(protocol=str(protocol_path), training_jobs=len(training),
                          eval_jobs=len(evaluations), frozen_files=len(paths))))


if __name__ == "__main__":
    main()
