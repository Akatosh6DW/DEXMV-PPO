"""Render paired first-case results and one explicitly labeled guided success."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/orca_collision_refit/O20_hic53_elliptic/cube.xml"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    output = root / "videos"
    output.mkdir(exist_ok=True)
    cases = []
    for method in ("ppo", "guided", "residual"):
        cases.append(dict(run=f"{method}_s901_nominal", episode=0,
                          selection="paired first training seed and first held-out initial state, not best episode"))
    # Supplement the paired panel, never substitute away its failed cases.
    for seed in (902,903):
        run = f"guided_s{seed}_nominal"
        rows = json.loads((root / "eval" / run / "step_000000000/episodes.json").read_text())
        hit = next((i for i, row in enumerate(rows[:4]) if row["success30"]), None)
        if hit is not None:
            cases.append(dict(run=run, episode=hit,
                              selection="explicit illustrative success from saved first four cases; not representative aggregate"))
            break
    reports = []
    for case in cases:
        directory = root / "eval" / case["run"] / "step_000000000"
        rows = json.loads((directory / "episodes.json").read_text())
        destination = output / f"{case['run']}_episode{case['episode']}"
        command = [sys.executable, str(ROOT / "scripts/render_orca_modes.py"), "--model", str(MODEL),
                   "--trace", str(directory / "preview_trajectories.npz"), "--episode", str(case["episode"]),
                   "--output", str(destination), "--azimuth", "225", "--elevation", "-18", "--distance", ".38",
                   "--lookat", ".035", ".025", ".245", "--stop-after-termination"]
        if not (destination / "report.json").exists():
            subprocess.run(command, cwd=ROOT, env=dict(os.environ,MUJOCO_GL="egl",OMP_NUM_THREADS="1"), check=True)
        report = json.loads((destination / "report.json").read_text())
        assert report["same_saved_states_both_modes"] and report["no_physics_resimulation"]
        for mode in ("stl","collision"):
            video = destination / f"{mode}.mp4"
            stream = json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-show_entries",
                                                        "stream=nb_read_frames,r_frame_rate,width,height", "-of","json",str(video)]))["streams"][0]
            assert int(stream["nb_read_frames"]) == report["frames"]
            assert stream["r_frame_rate"] == "20/1"
        reports.append(dict(case, metrics=rows[case["episode"]], output=str(destination), render=report))
    (output / "manifest.json").write_text(json.dumps(reports,indent=2)+"\n")
    print(json.dumps([dict(run=r["run"], episode=r["episode"], success=r["metrics"]["success30"], output=r["output"]) for r in reports]),flush=True)


if __name__ == "__main__":
    main()
