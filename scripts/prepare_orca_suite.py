"""Freeze S02 jobs before inspecting any new held-out policy results."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("artifacts/orca_benchmark_suite/S02")
BENCHMARKS = ("B0_nominal", "B1_position", "B2_yaw", "B3_friction", "B4_mass", "B5_push")
METHODS = ("ppo", "guided", "residual", "linear", "unrefined")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args()
    output = ROOT / OUTPUT
    output.mkdir(parents=True, exist_ok=False)
    jobs_dir = output / "jobs"
    jobs_dir.mkdir()
    jobs = []
    def add(name, **values):
        job = dict(output=str(OUTPUT / name), **values)
        path = jobs_dir / (name.replace("/", "__") + ".json")
        path.write_text(json.dumps(job, indent=2)+"\n")
        jobs.append(dict(name=name, path=str(path.relative_to(ROOT)), sha256=digest(path)))
    recipes = {m: dict(std=.7 if m == "ppo" else .3,
                       learning_rate=.00003 if m == "guided" else .0003) for m in METHODS}
    for seed in range(1001, 1011):
        for method in METHODS:
            add(f"train/{method}_s{seed}", method=method, seed=seed, steps=163840,
                eval_start=32000, eval_count=8, audit=False, **recipes[method])
    for benchmark in BENCHMARKS:
        for seed in range(1001, 1011):
            for method in (*METHODS, "bc"):
                source = "guided" if method == "bc" else method
                checkpoint = "bc_params.pkl" if method == "bc" else "params.pkl"
                add(f"eval/{benchmark}/{method}_s{seed}", method=method, seed=seed,
                    benchmark=benchmark, eval_start=40000, eval_count=200, audit=True, steps=0,
                    checkpoint=str(OUTPUT / f"train/{source}_s{seed}" / checkpoint), **recipes[source])
        for variant in ("video", "linear", "unrefined"):
            add(f"eval/{benchmark}/zero_{variant}", method="zero", variant=variant, seed=1001,
                benchmark=benchmark, eval_start=40000, eval_count=200, audit=True, steps=0)
    for seed in range(1001, 1011):
        add(f"eval/B0_nominal/half_residual_s{seed}", method="residual", seed=seed,
            benchmark="B0_nominal", residual_gain=.5, eval_start=40000, eval_count=200,
            audit=True, steps=0, checkpoint=str(OUTPUT / f"train/residual_s{seed}/params.pkl"))
    source_names = ("orca_suite_env.py", "run_orca_suite_job.py", "prepare_orca_suite.py",
                    "run_orca_suite.py", "test_orca_suite.py", "orca_acquisition_env.py",
                    "orca_native_physics.py", "orca_matched_env.py", "orca_hic53_env.py",
                    "orca_refit_env.py", "orca_residual_env.py", "orca_benchmark_env.py",
                    "orca_residual_controls.py", "orca_contact_forces.py", "orca_residual_metrics.py",
                    "orca_guidance_recheck.py", "train_orca_acquisition.py", "train_orca_matched.py",
                    "train_orca_comparison.py")
    paths = [ROOT / "scripts" / name for name in source_names]
    paths += [ROOT / "Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/orca_hand/grasp_cube.py",
              ROOT / "artifacts/orca_collision_refit/O20_hic53_elliptic/cube.xml",
              ROOT / "artifacts/orca_collision_refit/O20_hic53_elliptic/reference.npz",
              ROOT / "artifacts/orca_cube_acquisition/E01_flex50_refinement/candidate_reference.npz",
              ROOT / "artifacts/orca_cube_acquisition/D01_quality_demos/trajectories.npz",
              ROOT / "artifacts/orca_cube_acquisition/D01_quality_demos/dataset_manifest.json"]
    model = ROOT / "artifacts/orca_collision_refit/O20_hic53_elliptic/cube.xml"
    for mesh in ET.parse(model).getroot().findall("asset/mesh"):
        if mesh.get("file"):
            paths.append((model.parent / mesh.get("file")).resolve())
    protocol = dict(version="S02", date="2026-10-07", status="frozen_before_test_results",
                    output=str(OUTPUT), train_seeds=list(range(1001, 1011)),
                    dev_seeds=list(range(32000, 32008)), test_seeds=list(range(40000, 40200)),
                    training_steps=163840, benchmark_names=BENCHMARKS, recipes=recipes,
                    source_hashes={str(p.relative_to(ROOT)): digest(p) for p in paths}, jobs=jobs,
                    preregistration="docs/orca_benchmark_suite_protocol.md",
                    primary_contrasts=["residual-ppo", "residual-guided", "residual-zero_video",
                                       "residual-linear", "residual-unrefined"],
                    primary_metric="clean_success30 at native solver substep contacts",
                    secondary_metric="original success30, 20Hz state criteria",
                    video_selection="training seed1001, episode seed40000, every condition; failures retained",
                    limitations=["single video", "single object", "fixed wrist", "self collision disabled",
                                 "recipes selected in earlier M01 development; no new hyperparameter search",
                                 "no claim of matched total preparation compute", "not a hardware test"])
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    print(json.dumps(dict(jobs=len(jobs), train=50, eval=len(jobs)-50,
                         eval_episodes=(len(jobs)-50)*200, protocol=str(output / "protocol.json"))))


if __name__ == "__main__":
    main()
