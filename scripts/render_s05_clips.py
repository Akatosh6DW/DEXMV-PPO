"""Render S05 saved evaluation states with palm-facing and collision views."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("LP_NUM_THREADS", "4")

import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from export_grasp_presentation_videos import clip_endpoint
from inspect_orca_mesh_alignment import build_overlay


def s05_clip_endpoint(trace, success):
    """A failed pad grasp ends when an established valid hold is first lost."""
    result = clip_endpoint(trace, success)
    if success:
        return result
    established = np.maximum.accumulate(np.asarray(trace["hold"]) >= 4.)
    lost = np.flatnonzero(established & (np.asarray(trace["hold"]) <= 0)
                          & np.asarray(trace["active"]).astype(bool))
    if len(lost) and lost[0] < result[0]:
        return int(lost[0]), None, "首次丢失已建立的合格指腹抓持", "失败"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stills-only", action="store_true")
    parser.add_argument("--azimuth-offset", type=float, default=0.,
                        help="Optional oblique inspection angle relative to the palm camera")
    parser.add_argument("--distance", type=float, default=.36)
    parser.add_argument("--transparent-object", action="store_true",
                        help="Diagnostic stills only; reveal otherwise occluded finger contacts")
    parser.add_argument("--policy-label", help="Explicit filename label, e.g. BC-only for a zero-step guided checkpoint")
    parser.add_argument("--task-label", default="方块_S05", help="Task identifier used in output filenames")
    parser.add_argument("--header-label", default="S05", help="Task identifier shown in the video")
    args = parser.parse_args()
    if args.transparent_object and not args.stills_only:
        parser.error("Transparent object is for diagnostic stills only, not showcase videos")
    args.output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.run / "run_config.json").read_text())
    display_method = "BC only" if args.policy_label == "仅BC" else meta["method"]
    directory = args.run / f"step_{args.step:09d}"
    episode = json.loads((directory / "episodes.json").read_text())[args.index]
    with np.load(directory / "preview_trajectories.npz") as archive:
        trace = {k: archive[k][args.index] for k in archive.files}
    with np.load(directory / "initial_states.npz") as archive:
        initial = archive["qpos"][args.index]
    with np.load(meta["reference"]) as reference:
        center = reference["object_position"]
    model, _ = build_overlay(Path(meta["model"]), args.output / "overlay.xml")
    if args.transparent_object:
        cube_body = model.body("cube").id
        model.geom_rgba[model.geom_bodyid == cube_body, 3] = .25
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
    data = mujoco.MjData(model)
    end, _, reason, status = s05_clip_endpoint(trace, episode["success30"])
    # Infer the palm-facing side from the straight index's flexion direction;
    # then follow the actual wrist pose rather than assuming the original mount.
    palm = model.body("right_R-Carpals_8d1f1041").id
    index = model.body("right_I-FingerTipAssembly_ec49c16c").id
    data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]] = 0
    mujoco.mj_forward(model, data)
    palm_axis = data.xmat[palm].reshape(3, 3).T @ (-data.xmat[index].reshape(3, 3)[:, 0])
    data.qpos[:] = trace["qpos"][min(60, len(trace["qpos"]) - 1)]
    mujoco.mj_forward(model, data)
    offset = data.xmat[palm].reshape(3, 3) @ palm_axis
    offset += np.array([0., 0., .15])
    offset /= np.linalg.norm(offset)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = center
    camera.distance = args.distance
    camera.azimuth = np.rad2deg(np.arctan2(-offset[1], -offset[0])) + args.azimuth_offset
    # MuJoCo's free-camera offset is the negative viewing direction.
    camera.elevation = float(-np.rad2deg(np.arcsin(offset[2])))
    # Keep the camera above the floor even for a palm-down mounting.
    camera.elevation = min(camera.elevation, np.rad2deg(np.arcsin(min(.95, (center[2] - .05) / camera.distance))))
    still_frames = sorted({0, min(60, end + 1), min(140, end + 1), end + 1})
    panels = []
    with mujoco.Renderer(model, 720, 960) as renderer:
        for mode in ("stl", "collision"):
            option = mujoco.MjvOption()
            option.geomgroup[2] = mode == "stl"
            option.geomgroup[3] = mode == "collision"
            video = []
            frames = still_frames if args.stills_only else range(end + 2)
            for f in frames:
                i = max(0, min(f - 1, len(trace["qpos"]) - 1))
                data.qpos[:] = initial if f == 0 else trace["qpos"][i]
                data.qvel[:] = 0 if f == 0 else trace["qvel"][i]
                data.ctrl[:] = trace["ctrl"][i]
                data.mocap_pos[:] = trace["mocap"][i]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera, scene_option=option)
                picture = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(picture)
                draw.rectangle((0, 0, 960, 42), fill=(24, 26, 28))
                diagnostic = " | TRANSPARENT OBJECT: INSPECTION ONLY" if args.transparent_object else ""
                draw.text((12, 12), f"{args.header_label} | {display_method} | {mode} | t={f*.05:.2f}s | strict hold={trace['hold'][i]:.2f}s" + diagnostic, fill="white")
                if f in still_frames:
                    picture.save(args.output / f"{mode}_{f:04d}.png")
                    panels.append(picture.resize((480, 360)))
                if not args.stills_only:
                    video.append(np.array(picture))
            if video:
                label = "外观" if mode == "stl" else "碰撞几何"
                view = "手心" if args.azimuth_offset == 0 else f"斜向{args.azimuth_offset:g}度"
                policy_label = args.policy_label or {"residual": "视频参考加残差PPO", "guided": "BC加PPO",
                    "reference": "仅参考回放", "bc": "仅BC", "ppo": "纯PPO"}.get(meta["method"], meta["method"])
                mediapy.write_video(str(args.output / f"{args.task_label}_{policy_label}_{view}_{status}_{label}.mp4"), video, fps=20)
    sheet = Image.new("RGB", (480 * len(still_frames), 720))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % len(still_frames)) * 480, (i // len(still_frames)) * 360))
    sheet.save(args.output / "contact_sheet.jpg")
    (args.output / "render.json").write_text(json.dumps(dict(run=str(args.run), step=args.step,
        episode=episode, endpoint=end, reason=reason, transparent_object=args.transparent_object,
        camera=dict(azimuth=camera.azimuth,
        elevation=camera.elevation, distance=camera.distance, lookat=camera.lookat.tolist())), indent=2) + "\n")


if __name__ == "__main__":
    main()
