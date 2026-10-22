"""Freeze the complete three-method cylinder matrix after development."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/orca_cylinder_study"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--ppo-std", type=float, choices=(.3,.7), default=.7)
    args = p.parse_args()
    if (OUT / "protocol.json").exists():
        raise FileExistsError("A frozen protocol already exists")
    reference = str(args.reference.resolve().relative_to(ROOT))
    dataset = str(args.dataset.resolve().relative_to(ROOT))
    recipes = {"ppo": .0003, "guided": .0000075, "residual": .0003}
    standard_deviations = {"ppo":args.ppo_std,"guided":.3,"residual":.3,"bc":.3,"zero":.3}
    seeds, budgets = [41001, 41002, 41003], [163840, 655360]
    conditions = ["B0_nominal", "B1_position", "B5_push"]
    jobs = OUT / "jobs"
    jobs.mkdir(exist_ok=False)
    training, evaluation, hashes = [], [], {}

    def save(name, job, target):
        path = jobs / (name + ".json")
        path.write_text(json.dumps(job, indent=2)+"\n")
        rel = str(path.relative_to(ROOT))
        target.append(rel)
        hashes[rel] = digest(path)

    def base(method, seed):
        return dict(method=method, seed=seed, std=standard_deviations[method], reference=reference,
                    shaping=1., steps=0, eval_count=64, eval_start=110000)

    for seed in seeds:
        for method, lr in recipes.items():
            train_dir = f"artifacts/orca_cylinder_study/train/{method}/s{seed}"
            job = dict(base(method, seed), output=train_dir, steps=655360, evals=5,
                       eval_start=100000, eval_count=16, learning_rate=lr)
            if method == "guided":
                job.update(dataset=dataset, bc_updates=4000)
            save(f"train_{method}_{seed}", job, training)
            for budget in budgets:
                for condition in conditions:
                    job = dict(base(method, seed), benchmark=condition, budget=budget,
                               checkpoint=f"{train_dir}/step_{budget:09d}/params.pkl",
                               output=f"artifacts/orca_cylinder_study/eval/{condition}/{method}/s{seed}/b{budget}")
                    save(f"eval_{condition}_{method}_{seed}_{budget}", job, evaluation)
        for condition in conditions:
            job = dict(base("bc", seed), benchmark=condition, budget=0,
                       checkpoint=f"artifacts/orca_cylinder_study/train/guided/s{seed}/bc_params.pkl",
                       output=f"artifacts/orca_cylinder_study/eval/{condition}/bc/s{seed}/b0")
            save(f"eval_{condition}_bc_{seed}", job, evaluation)
    for condition in conditions:
        job = dict(base("zero", 41001), benchmark=condition, budget=0,
                   output=f"artifacts/orca_cylinder_study/eval/{condition}/zero/s41001/b0")
        save(f"eval_{condition}_zero", job, evaluation)
    if len(training) != 9 or len(evaluation) != 66:
        raise AssertionError("Unexpected matrix size")
    frozen = {}
    for path in (args.reference, args.dataset, args.dataset.parent / "dataset_manifest.json"):
        frozen[str(path.resolve().relative_to(ROOT))] = digest(path)
    # Freeze the loaded local training dependency set and the base model's meshes.
    collect = json.loads((args.dataset.parent / "run_config.json").read_text())
    for name in collect["source_sha256"]:
        path = ROOT / "scripts" / name
        frozen[str(path.relative_to(ROOT))] = digest(path)
    for name in ("prepare_orca_cylinder_study.py", "run_orca_cylinder_study.py"):
        path = ROOT / "scripts" / name
        frozen[str(path.relative_to(ROOT))] = digest(path)
    playground = ROOT / "Aero-Hand/sim_rl/mujoco_playground/mujoco_playground"
    runtime = list((playground / "_src/manipulation/orca_hand").glob("*.py"))
    runtime += [playground / name for name in ("__init__.py", "_src/mjx_env.py", "_src/wrapper.py")]
    for path in runtime:
        frozen[str(path.relative_to(ROOT))] = digest(path)
    source = ROOT / "artifacts/orca_collision_refit/O20_hic53_elliptic/cube.xml"
    frozen[str(source.relative_to(ROOT))] = digest(source)
    root = ET.parse(source).getroot()
    compiler = root.find("compiler")
    meshdir = source.parent / (compiler.get("meshdir", "") if compiler is not None else "")
    for asset in root.findall("asset/mesh"):
        path = (meshdir / asset.get("file")).resolve()
        frozen[str(path.relative_to(ROOT))] = digest(path)
    protocol = dict(version=1, frozen_before_training_and_test=True,
                    task="Fixed-base Orca, 53mm diameter/height cylinder, 35g, 40s episode",
                    primary="30s valid hold plus valid final frame; no post-release forbidden substep contact; depth<=3mm",
                    reference_origin="HIC20 cube tracking plus explicitly recorded cylinder simulator calibration",
                    train_seeds=seeds, budgets=budgets, test_seeds=list(range(110000,110064)),
                    conditions=conditions, main_condition="B0_nominal", selected_checkpoints="fixed budgets, never best-on-test",
                    methods=recipes, initial_std=standard_deviations, shared_shaping=1., bc_updates=4000,
                    hyperparameter_origin="Historical S03 learning rates; two equal-budget development std candidates per method; no test tuning",
                    interpretation="Three training seeds only; descriptive paired results, not proof of universal superiority",
                    reference=reference, dataset=dataset, train_jobs=training, eval_jobs=evaluation,
                    job_sha256=hashes, frozen_inputs=frozen, expected_training_runs=9,
                    expected_evaluation_jobs=66, expected_test_episodes=4224,
                    total_online_steps=5898240, extra_costs_excluded_from_online_budget=["reference calibration", "demonstration collection", "BC"])
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    print(json.dumps({k:protocol[k] for k in ("expected_training_runs", "expected_evaluation_jobs", "expected_test_episodes", "total_online_steps")}))


if __name__ == "__main__":
    main()
