"""Render prespecified first test cases; never choose only successful episodes."""

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/orca_cylinder_study"


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    output = OUT / "videos"
    output.mkdir(exist_ok=True)
    cases = []
    for path in protocol["eval_jobs"]:
        job = json.loads((ROOT / path).read_text())
        if job["seed"] != 41001 or job["budget"] not in (0,655360):
            continue
        if job["benchmark"] != "B0_nominal" and job["method"] != "residual":
            continue
        run = ROOT / job["output"]
        if not (run / "summary.json").exists():
            raise FileNotFoundError("Prespecified case incomplete: " + str(run))
        name = f"{job['benchmark']}_{job['method']}_s41001"
        destination = output / name
        trace = run / "step_000000000/preview_trajectories.npz"
        command = [sys.executable, str(ROOT / "scripts/render_orca_modes.py"),
                   "--model", str(run / "model.xml"), "--trace", str(trace), "--episode", "0",
                   "--output", str(destination), "--azimuth", "225", "--elevation", "-18",
                   "--distance", ".38", "--lookat", ".035", ".025", ".245"]
        if not (destination / "report.json").exists():
            if destination.exists():
                raise FileExistsError("Inspect partial render: " + str(destination))
            print("RENDER", name, flush=True)
            with (output / (name + ".log")).open("x") as stream:
                subprocess.run(command, cwd=ROOT, env=dict(os.environ,MUJOCO_GL="egl",OPENBLAS_NUM_THREADS="1"),
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
        report = json.loads((destination / "report.json").read_text())
        assert report["frames"] == 800 and report["episode"] == 0 and report["no_physics_resimulation"]
        assert report["physics_arrays_match"]
        for mode in ("stl", "collision"):
            probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                              "-count_frames", "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate,duration",
                              "-of", "json", str(destination / (mode + ".mp4"))]))["streams"][0]
            assert int(probe["nb_read_frames"]) == 800 and probe["r_frame_rate"] == "20/1"
            assert (probe["width"],probe["height"]) == (960,720)
            assert min(report["pixel_checks"][mode][x] for x in ("first_std","last_std")) > 5
        episode = json.loads((run / "step_000000000/episodes.json").read_text())[0]
        cases.append(dict(name=name, job=path, seed=episode["seed"], outcome=episode,
                          stl=f"{name}/stl.mp4", collision=f"{name}/collision.mp4", camera=report["camera"]))
    assert len(cases) == 7
    (output / "manifest.json").write_text(json.dumps(dict(complete=True,cases=cases,
           selection="Training seed41001, test seed110000, long-budget policies; no success-based selection"),indent=2)+"\n")
    lines=["# 圆柱固定案例回放", "", "每个方法固定训练种子41001和测试初态110000，保留成功和失败，不按效果挑视频。",
           "手心视角；从保存状态渲染，不重新仿真。终止后的画面仅作诊断，不计成功。", "",
           "| 案例 | 清洁成功 | 最长有效保持秒 | STL | 碰撞体 |", "|---|---|---:|---|---|"]
    for c in cases:
        lines.append(f"| {c['name']} | {c['outcome']['clean_success30']} | {c['outcome']['longest_hold_sec']:.2f} | "
                     f"[播放]({c['stl']}) | [播放]({c['collision']}) |")
    (output / "index.md").write_text("\n".join(lines)+"\n")
    print("VIDEOS_COMPLETE", len(cases)*2, flush=True)


if __name__ == "__main__":
    main()
