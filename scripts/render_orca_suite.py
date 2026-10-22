"""Render prespecified S02 test cases and verify both stored-state video modes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selections(protocol):
    for benchmark in protocol["benchmark_names"]:
        for method in ("ppo", "guided", "residual", "zero_video"):
            yield benchmark, method
    for method in ("bc", "linear", "unrefined", "zero_linear", "zero_unrefined", "half_residual"):
        yield "B0_nominal", method


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    output = root / "videos"
    output.mkdir(exist_ok=True)
    manifest, missing = [], []
    for benchmark, method in selections(protocol):
        name = method if method.startswith("zero_") else method + "_s1001"
        run = root / "eval" / benchmark / name
        trace = run / "step_000000000/preview_trajectories.npz"
        if not (run / "summary.json").exists():
            missing.append(str(run.relative_to(root)))
            continue
        episodes = json.loads((trace.parent / "episodes.json").read_text())
        assert episodes[0]["seed"] == 40000
        config = json.loads((run / "run_config.json").read_text())
        directory = output / (benchmark + "__" + name + "__case40000")
        command = [sys.executable, str(ROOT / "scripts/render_orca_modes.py"),
                   "--model", config["model"], "--trace", str(trace),
                   "--episode", "0", "--output", str(directory), "--azimuth", "225",
                   "--elevation", "-18", "--distance", ".38", "--lookat", ".035", ".025", ".245"]
        if not (directory / "report.json").exists():
            if directory.exists():
                raise FileExistsError("Inspect partial video before retry: " + str(directory))
            print("RENDER", benchmark, method, flush=True)
            with (output / (directory.name + ".log")).open("w") as log:
                subprocess.run(command, cwd=ROOT, env=dict(os.environ, MUJOCO_GL="egl"),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((directory / "report.json").read_text())
        assert report["trace_sha256"] == digest(trace)
        assert report["episode"] == 0 and report["frames"] == 800 and report["seconds"] == 40
        assert report["same_saved_states_both_modes"] and report["no_physics_resimulation"]
        videos = {}
        for mode in ("stl", "collision"):
            path = directory / (mode + ".mp4")
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate,duration",
                "-of", "json", str(path)], text=True))["streams"][0]
            assert int(probe["nb_read_frames"]) == 800
            assert probe["r_frame_rate"] == "20/1"
            assert (probe["width"], probe["height"]) == (960, 720)
            assert abs(float(probe["duration"]) - 40) < .01
            assert report["pixel_checks"][mode]["first_std"] > 5
            videos[mode] = dict(path=str(path.relative_to(root)), sha256=digest(path), probe=probe)
        manifest.append(dict(benchmark=benchmark, method=method,
                             train_seed=None if method.startswith("zero_") else 1001, test_seed=40000,
                             outcome=episodes[0], videos=videos, command=command, camera=report["camera"],
                             source_trace_sha256=digest(trace), run_config_sha256=digest(run / "run_config.json"),
                             note="Fixed case, including failures. Post-termination states are diagnostic only. Mass/friction/push conditions come from run_config, not video appearance."))
        (output / "manifest.json").write_text(json.dumps(dict(complete=False, cases=manifest), indent=2)+"\n")
    result = dict(complete=not missing, missing=missing, cases=manifest,
                  renderer_sha256=digest(ROOT / "scripts/render_orca_modes.py"),
                  selection_source_sha256=digest(Path(__file__)), protocol_sha256=digest(root / "protocol.json"))
    (output / "manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["# S02 固定案例视频", "", "学习策略使用训练种子1001，零修正参考无训练种子；测试初态40000。成功和失败均保留，不按效果挑选。",
             "手心视角；同一条保存轨迹分别显示STL和碰撞体。失败后的诊断尾段不计入成绩。", "",
             "| 条件 | 方法 | 原始/清洁成功 | STL | 碰撞体 |", "|---|---|---|---|---|"]
    for item in manifest:
        outcome = item["outcome"]
        links = ["[播放](" + str(Path(item["videos"][mode]["path"]).relative_to("videos")) + ")"
                 for mode in ("stl", "collision")]
        lines.append(f"| {item['benchmark']} | {item['method']} | "
                     f"{outcome['success30']}/{outcome['clean_success30']} | {links[0]} | {links[1]} |")
    lines += ["", f"状态：{'完成' if not missing else '部分完成'}，案例{len(manifest)}，缺失{len(missing)}。"]
    (output / "index.md").write_text("\n".join(lines)+"\n")
    if missing and not args.allow_partial:
        raise RuntimeError("Missing completed evaluations: " + str(missing))
    print(json.dumps(dict(complete=not missing, cases=len(manifest), missing=len(missing))))


if __name__ == "__main__":
    main()
