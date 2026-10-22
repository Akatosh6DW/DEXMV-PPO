#!/usr/bin/env python3
"""Render one saved rollout in collision and CAD modes, without resimulation."""

import argparse
import hashlib
import json
from pathlib import Path

import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from inspect_orca_mesh_alignment import build_overlay


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--azimuth", type=float, default=45)
    p.add_argument("--elevation", type=float, default=-18)
    p.add_argument("--distance", type=float, default=.36)
    p.add_argument("--lookat", type=float, nargs=3, default=[.065, .025, .245])
    p.add_argument("--stop-after-termination", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model, report = build_overlay(args.model, args.output / "visual_overlay.xml")
    with np.load(args.trace) as z:
        qpos = z["qpos"]
        batched = qpos.ndim == 3
        if batched:
            qpos = qpos[args.episode]
        mocap = z["mocap"] if "mocap" in z else np.zeros((len(qpos), 0, 3))
        if batched and "mocap" in z:
            mocap = mocap[args.episode]
        degrees = z["degrees"] if "degrees" in z else None
        errors = z["thread_error"] if "thread_error" in z else None
        if degrees is not None and batched:
            degrees, errors = degrees[args.episode], errors[args.episode]
        grasp = {}
        for key in ("hold", "contacts", "thumb", "forbidden", "alive", "cube_z", "spawn_z",
                    "released", "hold_sec", "success", "lift"):
            if key in z:
                grasp[key] = z[key][args.episode] if batched else z[key]
    if args.stop_after_termination and "alive" in grasp:
        stopped = np.flatnonzero(~np.asarray(grasp["alive"], dtype=bool))
        if len(stopped):
            end = int(stopped[0]) + 1
            qpos, mocap = qpos[:end], mocap[:end]
            grasp = {k: v[:end] for k, v in grasp.items()}
            if degrees is not None:
                degrees, errors = degrees[:end], errors[:end]
    if qpos.shape != (len(qpos), model.nq) or not np.isfinite(qpos).all():
        raise ValueError("Invalid saved qpos")
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = args.lookat
    camera.distance, camera.azimuth, camera.elevation = args.distance, args.azimuth, args.elevation
    option = mujoco.MjvOption()
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
    pixel_checks = {}
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    with mujoco.Renderer(model, 720, 960) as renderer:
        for mode in ("collision", "stl"):
            option.geomgroup[:] = 1
            option.geomgroup[2] = mode == "stl"
            option.geomgroup[3] = mode == "collision"
            first = last = None
            with mediapy.VideoWriter(str(args.output / f"{mode}.mp4"), shape=(720, 960), fps=20) as writer:
                for i, q in enumerate(qpos):
                    data.qpos[:] = q
                    if model.nmocap:
                        data.mocap_pos[:] = mocap[i]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera, scene_option=option)
                    frame = renderer.render().copy()
                    if degrees is not None:
                        overlay = Image.fromarray(frame)
                        draw = ImageDraw.Draw(overlay)
                        cap_status = {"released", "hold_sec", "success", "lift"}.issubset(grasp)
                        draw.rectangle((0, 0, 960, 92 if cap_status else 58), fill=(24, 26, 28))
                        thread = ("thread=RELEASED" if cap_status and grasp["released"][i]
                                  else f"thread error={abs(errors[i])*1000:.3f} mm")
                        label = f"{mode.upper()} | t={(i+1)/20:.2f}s | cap={degrees[i]:.1f} deg | {thread}"
                        draw.text((16, 16), label, font=font, fill=(240, 240, 240))
                        if cap_status:
                            status = (f"released={int(grasp['released'][i])} | lift={1000*grasp['lift'][i]:.1f} mm | "
                                      f"valid hold={grasp['hold_sec'][i]:.2f}s | success={int(grasp['success'][i])}")
                            draw.text((16, 54), status, font=font, fill=(240, 240, 240))
                        frame = np.asarray(overlay)
                    elif {"hold", "contacts", "thumb", "forbidden", "alive"}.issubset(grasp):
                        overlay = Image.fromarray(frame)
                        draw = ImageDraw.Draw(overlay)
                        active = bool(grasp["alive"][i])
                        has_height = "cube_z" in grasp and "spawn_z" in grasp
                        draw.rectangle((0, 0, 960, 114 if has_height else 84), fill=(24, 26, 28))
                        label = (f"{mode.upper()} | t={(i+1)/20:.2f}s | "
                                 f"continuous hold={grasp['hold'][i]:.2f}s | "
                                 f"contacts={int(grasp['contacts'][i])}")
                        status = (f"thumb={int(grasp['thumb'][i])} | "
                                  f"forbidden={int(grasp['forbidden'][i])} | "
                                  + ("ACTIVE" if active else "TERMINATED: diagnostic tail only"))
                        draw.text((16, 10), label, font=font, fill=(240, 240, 240))
                        draw.text((16, 46), status, font=font,
                                  fill=(240, 240, 240) if active else (255, 130, 110))
                        if has_height:
                            difference = 1000 * (grasp["cube_z"][i] - grasp["spawn_z"][i])
                            draw.text((16, 80), f"height change={difference:+.1f} mm | required > -10 mm", font=font,
                                      fill=(240, 240, 240) if difference > -10 else (255, 130, 110))
                        frame = np.asarray(overlay)
                    writer.add_image(frame)
                    if i == 0:
                        first = frame
                    last = frame
                    if i in (0, min(60, len(qpos)-1), len(qpos)-1):
                        mediapy.write_image(str(args.output / f"{mode}_{i:04d}.png"), frame)
            pixel_checks[mode] = dict(first_std=float(first.std()), last_std=float(last.std()),
                                     mean_frame_change=float(np.abs(last.astype(float)-first).mean()))
            if min(first.std(), last.std()) < 5:
                raise AssertionError("Blank render")
    report.update(trace=str(args.trace.resolve()), trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
                  renderer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  episode=args.episode, frames=len(qpos), fps=20, seconds=len(qpos)/20,
                  same_saved_states_both_modes=True, no_physics_resimulation=True,
                  stop_after_termination=args.stop_after_termination,
                  pixel_checks=pixel_checks,
                  camera=dict(azimuth=args.azimuth, elevation=args.elevation, lookat=args.lookat, distance=args.distance))
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
