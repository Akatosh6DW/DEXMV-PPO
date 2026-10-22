"""Freeze the post-pilot exploratory comparison before independent evaluation."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/orca_cube_acquisition/M01_comparison"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    recipes = dict(ppo=dict(std=.7, learning_rate=.0003),
                   guided=dict(std=.3, learning_rate=.00003),
                   residual=dict(std=.3, learning_rate=.0003),
                   bc=dict(std=.3, learning_rate=.00003))
    dataset = str(ROOT / "artifacts/orca_cube_acquisition/D01_quality_demos/trajectories.npz")
    jobs = []
    for seed in (901,902,903):
        for method in ("ppo", "guided", "residual", "bc"):
            options = dict(method=method, seed=seed, steps=0 if method=="bc" else 163840,
                           evals=5, eval_count=8, eval_start=12000, **recipes[method])
            if method in ("guided", "bc"):
                options["dataset"] = dataset
            jobs.append(dict(name=f"train/{method}_s{seed}", args=options))
    for jitter, scenario in ((.002,"nominal"), (.004,"offset4mm")):
        for seed in (901,902,903):
            for method in ("ppo", "guided", "residual", "bc"):
                jobs.append(dict(name=f"eval/{method}_s{seed}_{scenario}", args=dict(
                    method=method, seed=seed, steps=0, eval_start=20000, eval_count=64,
                    checkpoint=str(OUT / f"train/{method}_s{seed}/params.pkl"),
                    jitter=jitter, geometry_audit=True, **recipes[method])))
        jobs.append(dict(name=f"eval/zero_s901_{scenario}", args=dict(method="residual", seed=901,
                         steps=0, eval_start=20000, eval_count=64, jitter=jitter, geometry_audit=True,
                         **recipes["residual"])))
    sources = ("train_orca_acquisition.py", "orca_acquisition_env.py", "run_orca_acquisition_study.py",
               "train_orca_matched.py", "orca_native_physics.py", "orca_benchmark_env.py")
    protocol = dict(version="open_acquisition_v1_post_pilot", output=str(OUT.relative_to(ROOT)),
                    purpose="Exploratory limited-interaction-budget comparison, not universal algorithm ranking",
                    independent_training_seeds=[901,902,903], main_ppo_steps=163840,
                    dev_seeds=list(range(12000,12008)), test_seeds=list(range(20000,20064)),
                    source_video_count=1, source_sequence=20, source_frames=[30,95],
                    recipes=recipes, source_hashes={str(ROOT / "scripts" / n):hashlib.sha256((ROOT / "scripts" / n).read_bytes()).hexdigest() for n in sources},
                    raw_success="longest valid unsupported hold>=30s and valid final state at40s",
                    clean_success="raw success and no forbidden contacts after release and saved active-state penetration<=3mm",
                    controls="BC-only paired initialization and zero-residual refined reference",
                    total_test_episodes=26*64, jobs=jobs,
                    scope="No cap tasks. No held-out source-video generalization. All failures retained; final checkpoint, not development-best checkpoint.")
    path = OUT / "protocol.json"
    path.write_text(json.dumps(protocol, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
