"""Render all prespecified S03 palm-view cases, including failed episodes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_orca_s03_confirmation import OUT, ROOT, digest


def selections(jobs):
    for job in jobs:
        if job["seed"] != 21001 or job["budget"] not in (0, 655360):
            continue
        for mode in (("pushed_closed", "pushed_replay") if job.get("paired_replay") else ("evaluation",)):
            yield job, mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    protocol_path = OUT / "evaluation_protocol.json"
    plan = json.loads(protocol_path.read_text())
    selected = list(selections(plan["jobs"]))
    if len(selected) != 50:
        raise ValueError("Expected 50 prespecified cases")
    output = OUT / "videos"
    output.mkdir(exist_ok=True)
    manifest, missing = [], []
    for job, mode in selected:
        run = ROOT / job["output"]
        if not (run / "summary.json").exists():
            missing.append(job["output"]+"/"+mode)
            continue
        config = json.loads((run / "run_config.json").read_text())
        trace = run / mode / "preview_trajectories.npz"
        episode = json.loads((run / mode / "episodes.json").read_text())[0]
        if episode["seed"] != job["eval_start"]:
            raise ValueError("Unexpected selected initial state")
        name = f"{job['condition']}__{job['group']}__{mode}__case{episode['seed']}"
        directory = output / name
        command = [sys.executable, str(ROOT / "scripts/render_orca_modes.py"),
                   "--model", config["model"], "--trace", str(trace), "--episode", "0",
                   "--output", str(directory), "--azimuth", "225", "--elevation", "-18",
                   "--distance", ".38", "--lookat", ".035", ".025", ".245"]
        if not (directory / "report.json").exists():
            if directory.exists():
                raise FileExistsError("Inspect partial rendering: " + str(directory))
            print("RENDER", name, flush=True)
            with (output / (name+".log")).open("x") as log:
                subprocess.run(command, cwd=ROOT, env=dict(os.environ, MUJOCO_GL="egl"),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((directory / "report.json").read_text())
        if not (report["trace_sha256"] == digest(trace) and report["episode"] == 0
                and report["same_saved_states_both_modes"] and report["no_physics_resimulation"]):
            raise ValueError("Render identity mismatch")
        videos = {}
        for display in ("stl", "collision"):
            path = directory / (display+".mp4")
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate,duration", "-of", "json", str(path)],
                text=True))["streams"][0]
            if (int(probe["nb_read_frames"]) != 800 or probe["r_frame_rate"] != "20/1" or
                    (probe["width"], probe["height"]) != (960, 720) or abs(float(probe["duration"])-40) >= .01):
                raise ValueError("Invalid encoded video")
            if min(report["pixel_checks"][display][k] for k in ("first_std", "last_std")) <= 5:
                raise ValueError("Blank video")
            videos[display] = dict(path=str(path.relative_to(OUT)), sha256=digest(path), probe=probe)
        manifest.append(dict(condition=job["condition"], group=job["group"], mode=mode,
                             train_seed=None if job["method"] == "zero" else job["seed"],
                             test_seed=episode["seed"], outcome=episode, videos=videos,
                             run_config_sha256=digest(run / "run_config.json"),
                             source_trace_sha256=digest(trace), camera=report["camera"], command=command))
        (output / "manifest.json").write_text(json.dumps(dict(complete=False, cases=manifest), indent=2)+"\n")
    final = dict(complete=not missing, missing=missing, cases=manifest,
                 evaluation_protocol_sha256=digest(protocol_path), source_sha256=digest(Path(__file__)),
                 renderer_sha256=digest(ROOT / "scripts/render_orca_modes.py"))
    (output / "manifest.json").write_text(json.dumps(final, indent=2)+"\n")
    lines = ["# S03 固定案例视频", "", "手心视角，STL和碰撞体使用相同保存状态，不重新仿真。",
             "学习方法固定训练种子21001、各条件首个测试初态、长预算检查点；BC和零参考不额外训练。",
             "保留成功与失败，终止后的尾段只作诊断，不计入有效保持。", "",
             "| 条件 | 方法 | 模式 | 原始/清洁成功 | STL | 碰撞体 |", "|---|---|---|---|---|---|"]
    for c in manifest:
        links = ["[播放](" + str(Path(c["videos"][m]["path"]).relative_to("videos")) + ")" for m in ("stl", "collision")]
        lines.append(f"| {c['condition']} | {c['group']} | {c['mode']} | "
                     f"{c['outcome']['success30']}/{c['outcome']['clean_success30']} | {links[0]} | {links[1]} |")
    lines += ["", f"{'完成' if not missing else '部分完成'}：{len(manifest)}个案例、{2*len(manifest)}段视频；缺失{len(missing)}个案例。"]
    (output / "index.md").write_text("\n".join(lines)+"\n")
    if missing and not args.allow_partial:
        raise RuntimeError("Missing completed cases")
    print(json.dumps(dict(complete=not missing, cases=len(manifest), missing=len(missing))))


if __name__ == "__main__":
    main()
