"""Render every nominal seed's first held-out case, without success selection."""

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/orca_cylinder_raw_video_20261009"


def main():
    protocol = json.loads((OUT/"protocol.json").read_text())
    if not json.loads((OUT/"audit.json").read_text())["complete"]:
        raise ValueError("Full evaluation audit required")
    videos = OUT/"videos"
    videos.mkdir(exist_ok=True)
    cases = []
    for path in protocol["eval_jobs"]:
        job = json.loads((ROOT/path).read_text())
        if job["benchmark"] != "B0_nominal" or job["budget"] not in (0, 655360):
            continue
        run = ROOT/job["output"]
        name = f"formal_{job['method']}_s{job['seed']}"
        destination = videos/name
        command = [sys.executable, str(ROOT/"scripts/render_orca_modes.py"),
                   "--model", str(run/"model.xml"),
                   "--trace", str(run/"step_000000000/preview_trajectories.npz"),
                   "--episode", "0", "--output", str(destination),
                   "--azimuth", "225", "--elevation", "-18", "--distance", ".44",
                   "--lookat", ".06", ".035", ".245"]
        if not (destination/"report.json").exists():
            if destination.exists():
                raise FileExistsError("Inspect partial render: " + str(destination))
            print("RENDER", name, flush=True)
            with (videos/(name+".log")).open("x") as stream:
                subprocess.run(command, cwd=ROOT, env=dict(os.environ, MUJOCO_GL="egl", OPENBLAS_NUM_THREADS="1"),
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
        report = json.loads((destination/"report.json").read_text())
        assert report["physics_arrays_match"] and report["no_physics_resimulation"]
        assert report["frames"] == 800 and report["episode"] == 0
        for mode in ("stl", "collision"):
            stream = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate", "-of", "json",
                str(destination/(mode+".mp4"))]))["streams"][0]
            assert int(stream["nb_read_frames"]) == 800
            assert (stream["width"], stream["height"], stream["r_frame_rate"]) == (960, 720, "20/1")
            assert min(report["pixel_checks"][mode][key] for key in ("first_std", "last_std")) > 5
        episode = json.loads((run/"step_000000000/episodes.json").read_text())[0]
        cases.append(dict(name=name, method=job["method"], train_seed=job["seed"],
                          test_seed=episode["seed"], outcome=episode,
                          stl=f"{name}/stl.mp4", collision=f"{name}/collision.mp4"))
    assert len(cases) == 13
    (videos/"manifest.json").write_text(json.dumps(dict(complete=True, cases=cases,
           selection="All three training seeds at final budget; nominal held-out seed210000; no success filtering"), indent=2)+"\n")
    lines = ["# 从原始视频重做：固定案例视频", "",
             "正常位置测试的全部训练种子，固定测试初态210000，不按成功与否挑录像。每项有STL与碰撞体两种显示。",
             "网络在655360步时的固定权重；仅BC与仅参考没有在线PPO步数。",
             "终止后画面带有诊断尾段标记，不计成功。视频来自保存的仿真状态，不重新模拟。", "",
             "| 方法 | 训练种子 | 清洁30秒成功 | 最长有效保持秒 | STL | 碰撞体 |",
             "|---|---:|---|---:|---|---|"]
    for case in cases:
        ep = case["outcome"]
        lines.append(f"| {case['method']} | {case['train_seed']} | {ep['clean_success30']} | "
                     f"{ep['longest_hold_sec']:.2f} | [播放]({case['stl']}) | [播放]({case['collision']}) |")
    (videos/"index.md").write_text("\n".join(lines)+"\n")
    print("VIDEOS_COMPLETE", 2*len(cases), flush=True)


if __name__ == "__main__":
    main()
