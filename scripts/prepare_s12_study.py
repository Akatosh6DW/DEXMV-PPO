"""Freeze the V2 study before held-out evaluation, including executable job arguments."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import shutil

from s12_soft import SoftConfig, BASE, REFERENCE, scene_xml

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = SoftConfig(timestep=.0002, hand_offset_z=.003)
    dataset = ROOT/"artifacts/orca_benchmark_suite/S12/V2_D01_raise3mm/trajectories.npz"
    manifest = dataset.parent/"dataset_manifest.json"
    source_files = [BASE, REFERENCE, dataset, manifest,
        ROOT/"artifacts/orca_benchmark_suite/S09/inputs/bank_manifest.json",
        ROOT/"artifacts/orca_cylinder_s02/source/selection96/selected_segment.npz",
        ROOT/"artifacts/orca_cylinder_s02/source/selection96/selection.json"]
    code = [ROOT/"scripts"/f for f in ("s12_soft.py", "orca_s12_env.py", "s12_physics_worker.py",
        "s12_physics_pool.py", "run_s12_job.py", "evaluate_s12_checkpoint.py",
        "orca_native_physics.py", "orca_s10_env.py", "train_orca_comparison.py", "train_orca_matched.py")]
    inputs = {str(path.relative_to(ROOT)):digest(path) for path in source_files+code}
    sources = a.output/"sources"
    sources.mkdir()
    for path in code:
        shutil.copyfile(path, sources/path.name)
    (a.output/"model.xml").write_text(scene_xml(cfg))
    seeds = [25211,25212,25213]
    jobs = []
    for seed in seeds:
        for method in ("ppo", "guided", "residual"):
            output = a.output/"training"/f"{method}_s{seed}"
            argv = [sys.executable, str(ROOT/"scripts/run_s12_job.py"), "--output", str(output),
                "--method", method, "--seed", str(seed), "--steps", "51200", "--evals", "6",
                "--eval-start", "251010", "--eval-count", "8", "--hand-offset-z", ".003",
                "--physics-workers", "4", "--std", ".05", "--learning-rate", ".00003"]
            if method=="guided":
                argv += ["--dataset", str(dataset), "--bc-phase-only", "--bc-updates", "10000"]
            jobs.append(dict(name=output.name, method=method, seed=seed, argv=argv, output=str(output)))
    plan = dict(version="S12-V2-confirmation-v1", status="frozen_before_test", config=asdict(cfg),
        model_sha256=digest(a.output/"model.xml"), inputs=inputs, train_seeds=seeds,
        online_steps=51200, num_envs=8, reward="S12-R0", jitter=.002,
        train_evaluation_seeds=list(range(251010,251018)),
        demonstrations=dict(attempted_seeds=list(range(250100,250108)),
            selected_seeds=json.loads(manifest.read_text())["seeds"],
            unique_human_motion_sources=1, source="processed DexMV seq96, frames182..271, S09 template0"),
        selection="last checkpoint at 51200; no held-out checkpoint selection",
        primary="strict success15; separately report physical retention and all quality metrics",
        conditions={"nominal":dict(seeds=list(range(260000,260064))),
            "position5mm":dict(seeds=list(range(261000,261016))),
            "soft5kpa":dict(seeds=list(range(262000,262016))),
            "stiff20kpa":dict(seeds=list(range(263000,263016))),
            "fine":dict(seeds=list(range(264000,264008)))},
        methods=["ppo", "guided", "residual", "reference"], diagnostics=["bc_initial"],
        statistics="paired initial states; report individual training seeds and hierarchical bootstrap; reference is one deterministic controller, not three independent trained policies",
        limits=["finite-budget study, not asymptotic PPO performance", "one horizontal cylinder geometry",
            "simulation state, not raw pixels", "ideal shared Cartesian hand path",
            "unmeasured ideal elastic material", "fine-mesh contact classification sensitivity"],
        jobs=jobs)
    file = a.output/"protocol.json"
    file.write_text(json.dumps(plan, indent=2)+"\n")
    print(json.dumps(dict(protocol=str(file), sha256=digest(file), jobs=len(jobs)), indent=2))


if __name__ == "__main__":
    main()
