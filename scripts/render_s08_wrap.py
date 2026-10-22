"""Small palm-view wrap showcase and force plots from audited native traces."""

import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("LP_NUM_THREADS", "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from diagnose_orca_grasp import model_indices
from inspect_orca_mesh_alignment import build_overlay
from prepare_cylinder_s02_source import digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/orca_benchmark_suite/S08_wrap_refined"
DELIVERY = ROOT / "deliverables/S08_圆柱包覆可行性_2026-10-12"


def main():
    global BASE, DELIVERY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position-only", action="store_true")
    args = parser.parse_args()
    if args.position_only:
        BASE = ROOT / "artifacts/orca_benchmark_suite/S08_wrap_position"
        DELIVERY = ROOT / "deliverables/S08_纵向仅改位置_2026-10-12"
        audit = json.loads((BASE / "validation_audit.json").read_text())
        assert audit["identical_replay"] and audit["trace_criteria_checked"]
        source = json.loads((BASE / "report.json").read_text())
        assert source["warning_free"] and source["source_unchanged"]
        folder, trace_folder = BASE, BASE / "position_only"
        report = source["position_only"]
        direction, title = "纵向包覆_仅改初始位置", "Longitudinal position-only wrap"
    else:
        audit = json.loads((BASE / "validation_audit.json").read_text())
        assert all(r["warning_free"] and r["identical_replay"] for r in audit)
        folder = trace_folder = BASE / "transverse"
        report = json.loads((folder / "report.json").read_text())
        direction, title = "横向包覆", "Transverse wrap"
    DELIVERY.mkdir(exist_ok=False)
    with np.load(folder / "reference.npz") as z:
        packet = dict(z)
    selected = [next((r for r in report["tests"] if r["success30"] == success), None) for success in (True, False)]
    if args.position_only:
        # Show the first actual drop instead of a 40-second non-qualifying hold.
        selected[1] = next((r for r in report["tests"] if not r["success30"] and r["duration_sec"] < 40), selected[1])
    selected = [r for r in selected if r is not None]
    m, _ = build_overlay(folder / "model.xml", DELIVERY / "overlay.xml")
    m.vis.global_.offwidth, m.vis.global_.offheight = 960, 720
    d = mujoco.MjData(m)
    qids, oid, _ = model_indices(m)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = packet["object_position"]-[0, 0, .01 if args.position_only else .025]
    camera.distance, camera.azimuth, camera.elevation = .38 if args.position_only else .34, -105, -12
    index, panels = [], []
    with mujoco.Renderer(m, 720, 960) as renderer:
        for case in selected:
            with np.load(trace_folder / f"trace_{case['seed']}.npz") as z:
                trace = dict(z)
            end = int(np.flatnonzero(trace["hold"] >= 4)[0]) if case["success30"] else len(trace["hold"])-1
            rng = np.random.default_rng(case["seed"])
            spawn = packet["object_position"]+rng.uniform(-.002,.002,3)*[1,1,0]
            initial_hand = np.clip(packet["initial_ctrl"]+rng.normal(0,.002,17), *m.actuator_ctrlrange.T)
            for mode in ("外观", "碰撞几何"):
                opt = mujoco.MjvOption()
                opt.geomgroup[2], opt.geomgroup[3] = mode == "外观", mode == "碰撞几何"
                frames = []
                for frame in range(end+2):
                    i = max(0, frame-1)
                    d.qpos[:], d.qvel[:], d.ctrl[:], d.mocap_pos[:] = trace["qpos"][i], trace["qvel"][i], trace["ctrl"][i], trace["mocap"][i]
                    if frame == 0:
                        d.qpos[qids], d.qpos[oid:oid+7] = initial_hand, np.r_[spawn, packet["object_quaternion"]]
                        d.qvel[:], d.ctrl[:] = 0, packet["initial_ctrl"]
                    mujoco.mj_forward(m, d)
                    renderer.update_scene(d, camera, scene_option=opt)
                    picture = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(picture)
                    draw.rectangle((0,0,960,42), fill=(24,26,28))
                    draw.text((10,12), f"S08 palm wrap | geometry reference, NOT PPO | t={frame*.05:.2f}s | hold={0 if frame==0 else trace['hold'][i]:.2f}s", fill="white")
                    frames.append(np.asarray(picture))
                    if frame in (0, end+1):
                        picture.save(DELIVERY / f"{case['seed']}_{mode}_{frame:04d}.png")
                        panels.append(picture.resize((480,360)))
                status = "成功" if case["success30"] else "失败"
                path = DELIVERY / f"圆柱_S08_{direction}_几何参考非PPO_初态{case['seed']}_手心_{status}_{mode}.mp4"
                mediapy.write_video(str(path), frames, fps=20)
                subprocess.run(["ffmpeg","-v","error","-i",str(path),"-f","null","-"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                duration = float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1",str(path)], text=True))
                index.append(dict(file=path.name, seed=case["seed"], success=case["success30"], hold_sec=case["hold_sec"], duration_sec=duration, sha256=digest(path), decoded=True))
    sheet = Image.new("RGB",(1920,720))
    for i,picture in enumerate(panels):
        sheet.paste(picture, ((i%4)*480,(i//4)*360))
    sheet.save(DELIVERY / "contact_sheet.jpg")
    fig, axes = plt.subplots(1,2,figsize=(12,4),layout="constrained")
    axes[0].bar([str(r["seed"])[-3:] for r in report["tests"]], [r["hold_sec"] for r in report["tests"]], color="#278b79")
    axes[0].axhline(30,ls="--",color="black")
    axes[0].set(title=f"{title}: {report['successes']}/{report['count']}",xlabel="Test seed suffix",ylabel="Continuous wrap hold (s)")
    with np.load(trace_folder / f"trace_{selected[0]['seed']}.npz") as z:
        loads=z["loads"]
    for i,name in enumerate(("Palm","Thumb","Index","Middle","Ring","Pinky")):
        axes[1].plot(np.arange(1,len(loads)+1)*.05,loads[:,i],label=name)
    axes[1].set(title="Minimum curved-side load per 50ms interval",xlabel="Time (s)",ylabel="Normal load (N)",xlim=(0,10))
    axes[1].legend(fontsize=8)
    fig.savefig(BASE / "wrap_results.png",dpi=160)
    plt.close(fig)
    (DELIVERY / "来源与校验.json").write_text(json.dumps(index,indent=2,ensure_ascii=False)+"\n")
    lines=[f"# 圆柱{direction}：小规模可行性视频", "", f"这是几何求解后的参考回放，不是BC、PPO或残差训练成果。{report['count']}个新初态中{report['successes']}个通过严格30秒包覆检查；不能宣称已经完成学习方法对比。", "", "成功展示从起手到连续包覆4秒，完整轨迹实际验证40秒；失败在掉落终止处结束（若未掉落但判据不合格，则保留至评估结束）。外观/碰撞几何各一版。", "", "| 视频 | 完整评估成功 | 严格保持 | 时长 |", "|---|---|---:|---:|"]
    for r in index:
        lines.append(f"| [{r['file']}]({r['file']}) | {r['success']} | {r['hold_sec']:.2f}s | {r['duration_sec']:.2f}s |")
    lines += ["", f"详细说明：主工程`docs/experiments/S08/README.md`；原始结果：`{BASE.relative_to(ROOT)}/`。"]
    if args.position_only:
        lines += ["", "展示第一条成功83200和第一条实际掉落83215；另外4条未掉落但包覆判据不合格的失败轨迹保存在原始数据中。不是所有失败都等于物体落地。"]
    (DELIVERY / "目录.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(videos=len(index), output=str(DELIVERY)),ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
