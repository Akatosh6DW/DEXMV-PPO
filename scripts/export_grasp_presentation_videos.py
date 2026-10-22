"""Export palm-view presentation clips from immutable saved evaluation states."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("LP_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/grasp-presentation-mpl")

import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from inspect_orca_mesh_alignment import build_overlay

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "deliverables/抓取展示视频_2026-10-12"
WORK = ROOT / "artifacts/presentation_clips_20261012"
STUDIES = (
    ("方块_S02", "artifacts/orca_benchmark_suite/S02"),
    ("方块_S04", "artifacts/orca_benchmark_suite/S04"),
    ("圆柱_S02", "artifacts/orca_cylinder_s02"),
)
METHODS = {"ppo": "纯PPO", "guided": "BC加PPO", "residual": "视频参考加残差PPO"}
MODES = {"stl": "外观网格", "collision": "碰撞几何"}
CAMERA = dict(azimuth=225, elevation=-18, distance=.40, lookat=[.035, .025, .245])
DT = .05


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def four_second_endpoint(trace):
    valid = ((trace["hold"] > 0) & trace["alive"].astype(bool)
             & trace["active"].astype(bool) & (trace["released"] > 0)
             & (trace["thumb"] > 0) & (trace["contacts"] >= 2)
             & (trace["forbidden"] == 0) & (trace["substep_forbidden"] == 0)
             & (trace["substep_penetration"] <= .003))
    start = None
    for i, good in enumerate(valid):
        if not good:
            start = None
        elif start is None:
            start = i
        elif i - start >= 80:
            return i, start
    return None, None


def clip_endpoint(trace, success):
    if success:
        end, start = four_second_endpoint(trace)
        if end is None:
            raise ValueError("Reported success lacks four seconds of clean saved holding")
        return end, start, "稳定抓持连续满4秒", "成功"
    stopped = np.flatnonzero(~trace["alive"].astype(bool))
    audit_bad = np.flatnonzero((trace["active"] > 0) & (trace["released"] > 0)
                              & ((trace["substep_forbidden"] > 0) | (trace["substep_penetration"] > .003)))
    failures = []
    if len(stopped):
        failures.append((int(stopped[0]), "回合失败终止"))
    if len(audit_bad):
        failures.append((int(audit_bad[0]), "撤托后接触审计不合格"))
    if failures:
        end, reason = min(failures)
    else:
        end, reason = len(trace["qpos"]) - 1, "达到回合上限但未通过30秒标准"
    return end, None, reason, "失败"


def read_episode(path, index):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key][index] for key in z.files}


def prepare():
    cases = []
    for study, relative in STUDIES:
        for method in METHODS:
            base = ROOT / relative / "eval"
            runs = sorted((base / method).glob("s*")) if study == "方块_S04" else sorted(
                (base / "B0_nominal").glob(method + "_s*"))
            candidates = []
            for run in runs:
                trace_dir = run / ("evaluation" if study == "方块_S04" else "step_000000000")
                source = trace_dir / "preview_trajectories.npz"
                if not source.is_file():
                    continue
                rows = json.loads((trace_dir / "episodes.json").read_text())
                with np.load(source, allow_pickle=False) as z:
                    count = z["qpos"].shape[0]
                for index, row in enumerate(rows[:count]):
                    candidates.append((not bool(row["clean_success30"]), run, index, row, source))
            if not candidates:
                raise FileNotFoundError(f"No saved evaluation states: {study}/{method}")
            # A declared showcase selection, not a replacement for benchmark statistics.
            candidates.sort(key=lambda c: (c[0], str(c[1]), c[3]["seed"]))
            _, run, index, row, source = candidates[0]
            trace = read_episode(source, index)
            end, start, reason, status = clip_endpoint(trace, bool(row["clean_success30"]))
            config = json.loads((run / "run_config.json").read_text())
            model = Path(config["model"])
            if not model.is_absolute():
                model = ROOT / model
            if study == "圆柱_S02":
                exported = ROOT / relative / "videos/scenes/B0_nominal.xml"
                if exported.exists():
                    model = exported
            with np.load(source.parent / "initial_states.npz", allow_pickle=False) as z:
                assert int(z["seeds"][index]) == row["seed"]
            train_seed = int(config["seed"])
            base_name = f"{study}__{METHODS[method]}__手心__{status}__训练{train_seed}_初态{row['seed']}"
            cases.append(dict(id=study + "__" + method, study=study, method=method,
                              train_seed=train_seed, test_seed=row["seed"], episode=index,
                              source_trace=str(source.relative_to(ROOT)), trace_sha256=digest(source),
                              initial_states=str((source.parent / "initial_states.npz").relative_to(ROOT)),
                              initial_states_sha256=digest(source.parent / "initial_states.npz"),
                              model=str(model.relative_to(ROOT)), model_sha256=digest(model),
                              run_config=str((run / "run_config.json").relative_to(ROOT)),
                              run_config_sha256=digest(run / "run_config.json"),
                              outcome=row, endpoint=end, hold_start=start, end_reason=reason,
                              status=status, frames=end + 2, simulation_end_sec=(end + 1)*DT,
                              seconds=(end + 2)*DT, camera=CAMERA,
                              candidates_considered=len(candidates),
                              saved_clean_successes=sum(not c[0] for c in candidates),
                              videos={mode: base_name + "__" + label + ".mp4" for mode, label in MODES.items()}))
    return cases


def load_scene(case):
    directory = WORK / "scenes"
    directory.mkdir(parents=True, exist_ok=True)
    overlay_path = directory / (case["model_sha256"][:16] + ".xml")
    model, report = build_overlay(ROOT / case["model"], overlay_path)
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = CAMERA["lookat"]
    camera.distance, camera.azimuth, camera.elevation = CAMERA["distance"], CAMERA["azimuth"], CAMERA["elevation"]
    return model, data, camera, report


def encode_overlay(raw, case, mode, time_sec, hold):
    frame = Image.new("RGB", (960, 720), (24, 26, 28))
    frame.paste(Image.fromarray(raw), (0, 96))
    draw = ImageDraw.Draw(frame)
    draw.text((14, 6), f"{case['study']} | {METHODS[case['method']]} | 手心 | {MODES[mode]}",
              font=FONT, fill=(245, 245, 245))
    draw.text((14, 36), f"仿真时间 {time_sec:.2f}s   连续有效抓持 {hold:.2f}s   原30秒测试：{case['status']}",
              font=SMALL_FONT, fill=(220, 234, 226) if case["status"] == "成功" else (255, 170, 140))
    draw.text((14, 65), f"训练种子 {case['train_seed']} / 初态 {case['test_seed']}   展示案例，不代表平均成功率",
              font=SMALL_FONT, fill=(195, 199, 202))
    return np.asarray(frame)


def states(case):
    trace = read_episode(ROOT / case["source_trace"], case["episode"])
    with np.load(ROOT / case["initial_states"], allow_pickle=False) as z:
        initial = z["qpos"][case["episode"]].copy()
    return trace, initial


def set_frame(model, data, trace, initial, frame):
    if frame == 0:
        data.qpos[:] = initial
        data.qvel[:] = 0
        data.mocap_pos[:] = trace["mocap"][0]
    else:
        i = frame - 1
        data.qpos[:] = trace["qpos"][i]
        data.qvel[:] = trace["qvel"][i]
        data.ctrl[:] = trace["ctrl"][i]
        data.mocap_pos[:] = trace["mocap"][i]
    # Compute geometry only; never advance dynamics or rerun the controller.
    mujoco.mj_forward(model, data)


def preview(cases):
    panels = []
    for case in cases:
        model, data, cam, _ = load_scene(case)
        trace, initial = states(case)
        with mujoco.Renderer(model, 624, 960) as renderer:
            for mode in MODES:
                option = mujoco.MjvOption()
                option.geomgroup[2] = mode == "stl"
                option.geomgroup[3] = mode == "collision"
                for f in (0, min(case["endpoint"]+1, 61), case["endpoint"]+1):
                    set_frame(model, data, trace, initial, f)
                    renderer.update_scene(data, cam, scene_option=option)
                    raw = renderer.render().copy()
                    assert raw.std() > 5
                    img = Image.fromarray(encode_overlay(raw, case, mode, f*DT,
                                          0. if f == 0 else float(trace["hold"][f-1])))
                    img.save(WORK / (case["id"] + f"__{mode}__{f:04d}.png"))
                    panels.append(img.resize((480, 360)))
    sheet = Image.new("RGB", (1440, 360 * ((len(panels)+2)//3)), (255, 255, 255))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i%3)*480, (i//3)*360))
    sheet.save(WORK / "preview.jpg")
    print(json.dumps(dict(preview=str(WORK / "preview.jpg"), panels=len(panels))), flush=True)


def probe_video(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate,duration", "-of", "json", str(path)
    ], text=True))["streams"][0]


def render_case(case):
    report_path = WORK / (case["id"] + "__render.json")
    if report_path.exists():
        prior = json.loads(report_path.read_text())
        assert prior["case"] == case
        for mode, v in prior["videos"].items():
            assert digest(OUTPUT / case["videos"][mode]) == v["sha256"]
        return prior
    model, data, cam, scene_report = load_scene(case)
    trace, initial = states(case)
    report = dict(case=case, scene_check=scene_report, no_physics_resimulation=True, videos={})
    with mujoco.Renderer(model, 624, 960) as renderer:
        for mode in MODES:
            path = OUTPUT / case["videos"][mode]
            if path.exists():
                raise FileExistsError("Unverified existing clip: " + str(path))
            option = mujoco.MjvOption()
            option.geomgroup[2] = mode == "stl"
            option.geomgroup[3] = mode == "collision"
            first = last = None
            pixel_stds = []
            with mediapy.VideoWriter(str(path), shape=(720,960), fps=20, crf=19,
                                     ffmpeg_args=["-threads", "2", "-movflags", "+faststart"]) as writer:
                for f in range(case["frames"]):
                    set_frame(model, data, trace, initial, f)
                    renderer.update_scene(data, cam, scene_option=option)
                    raw = renderer.render().copy()
                    pixel_stds.append(float(raw.std()))
                    if f == 0:
                        first = raw.copy()
                    last = raw.copy()
                    frame = encode_overlay(raw, case, mode, f*DT, 0. if f == 0 else float(trace["hold"][f-1]))
                    writer.add_image(frame)
                    if f in (0, case["frames"]-1):
                        Image.fromarray(frame).save(WORK / (case["id"] + f"__{mode}__export_{f:04d}.png"))
            assert min(pixel_stds) > 5
            movement = float(np.abs(last.astype(float)-first.astype(float)).mean())
            assert movement > .1
            probe = probe_video(path)
            assert int(probe["nb_read_frames"]) == case["frames"]
            assert probe["r_frame_rate"] == "20/1"
            assert (probe["width"],probe["height"]) == (960,720)
            assert abs(float(probe["duration"]) - case["seconds"]) < .01
            report["videos"][mode] = dict(sha256=digest(path), probe=probe,
                                           min_scene_pixel_std=min(pixel_stds), mean_scene_change=movement)
            print(json.dumps(dict(done=path.name, seconds=case["seconds"]), ensure_ascii=False), flush=True)
    assert digest(ROOT / case["source_trace"]) == case["trace_sha256"]
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    return report


def write_index(cases, reports):
    lines = ["# 抓取视频目录", "", "日期：2026-10-12。全部MP4就在本文件夹，可以整文件夹移动，不依赖原工程播放。",
             "", "## 范围与观看说明", "",
             "- 方块S02、方块S04、圆柱S02；每组仅纯PPO、BC后PPO、视频参考加残差PPO。",
             "- 仅手心视角，每个案例有外观网格和实际碰撞几何两版，共18段。",
             "- 成功案例：从t=0的原始初态开始，截到撤托后连续有效且接触合格的抓持满4秒。",
             "- 失败案例：不在4秒截断，保留到失败终止或首次撤托后接触审计不合格；没有明确终止则保留到回合上限并标为未达标。",
             "- 以20 FPS原速播放，不加速、不定格、不重新仿真。同案例两版使用完全相同状态和截点。",
             "- 外观版是现有CAD网格与材质颜色，不是新生成的照片贴图；碰撞体并不全是长方体，还包括胶囊等实际接触代理。",
             "- 这是展示集：在每种方法已经保存的预览轨迹中，按训练种子、初态顺序优先取首个通过原30秒清洁标准的案例。",
             "- 没有成功预览则取首个已有案例，明确保留失败。选例不替代原实验统计，不可用这9个案例计算总体成功率。",
             "- S04复用S03的163840步模型；圆柱按原S02配方重新训练。没有额外训练或修改权重。", "",
             "## 播放入口", "", "| 任务/实验 | 策略 | 原30秒结果 | 片长 | 停止原因 | 外观网格 | 碰撞几何 |",
             "|---|---|---|---:|---|---|---|"]
    done = {r["case"]["id"] for r in reports}
    for c in cases:
        links = [f"[播放]({c['videos'][mode]})" if c["id"] in done else "待导出" for mode in MODES]
        lines.append(f"| {c['study']} | {METHODS[c['method']]} | {c['status']} | {c['seconds']:.2f}s | {c['end_reason']} | {' | '.join(links)} |")
    lines += ["", "## 案例来源", "", "| 任务 | 策略 | 训练种子 | 初态 | 原完整回合最长保持 | 保存候选中30秒成功数/总数 |",
              "|---|---|---:|---:|---:|---:|"]
    for c in cases:
        lines.append(f"| {c['study']} | {METHODS[c['method']]} | {c['train_seed']} | {c['test_seed']} | {c['outcome']['longest_hold_sec']:.2f}s | {c['saved_clean_successes']}/{c['candidates_considered']} |")
    lines += ["", "圆柱BC后PPO正常条件的正式总体成功率为0%；若短暂抓住，仍按失败片展示到失败，不包装为成功。",
              "完整来源、时间截点和视频SHA256见 `manifest.json`。其中源路径仅用于追溯，播放不需要它们。",
              "时长比末帧仿真时间多0.05秒是因为包含了t=0的初始帧，每帧显示0.05秒，不是额外保持时间。",
              "禁接触/穿透只保存每控制步汇总，首次审计失败时间定位到50ms控制帧，不声称精确定位到5ms子步。",
              "", f"当前完成：{len(reports)*2}/18 段。", ""]
    (OUTPUT / "目录.md").write_text("\n".join(lines))
    (OUTPUT / "manifest.json").write_text(json.dumps(dict(
        complete=len(reports)==len(cases), cases=cases, reports=reports,
        renderer_sha256=digest(Path(__file__)), backend="Mesa llvmpipe EGL software rendering",
        no_training=True, no_physics_resimulation=True), ensure_ascii=False, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--case", help="Render/preview a single case ID")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    cases = prepare()
    (WORK / "plan.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps([dict(id=c["id"], status=c["status"], seconds=c["seconds"],
                           seed=c["train_seed"], test=c["test_seed"]) for c in cases], ensure_ascii=False), flush=True)
    selected = [c for c in cases if not args.case or c["id"] == args.case]
    if not selected:
        raise ValueError("Unknown case")
    if args.preview:
        preview(selected)
        return
    reports = []
    write_index(cases, reports)
    for case in selected:
        t = time.monotonic()
        reports.append(render_case(case))
        write_index(cases, reports)
        print(json.dumps(dict(case=case["id"], render_seconds=time.monotonic()-t), ensure_ascii=False), flush=True)


FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 24)
SMALL_FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 20)

if __name__ == "__main__":
    main()
