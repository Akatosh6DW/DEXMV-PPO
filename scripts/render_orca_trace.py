#!/usr/bin/env python3
"""Render saved Orca native or MJX trajectories without changing physics."""

import argparse
import json
from pathlib import Path

import mediapy
import mujoco
import numpy as np

from diagnose_orca_grasp import SCENE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--azimuth", type=float, default=125)
    p.add_argument("--snapshots-only", action="store_true")
    args = p.parse_args()
    if args.trace.suffix == ".npz":
        with np.load(args.trace) as z:
            qs = z["qpos"][args.episode]
            ms = z["mocap"][args.episode]
    else:
        trace = json.loads(args.trace.read_text())["trace"]
        qs = np.array([s["qpos"] for s in trace])
        ms = np.array([s["mocap_pos"] for s in trace])
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 720
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [.065, .025, .245]
    cam.distance = .36
    cam.azimuth = args.azimuth
    cam.elevation = -18
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, 720, 960) as renderer:
        if args.snapshots_only:
            for i in (0, 20, 60, min(600, len(qs) - 1)):
                data.qpos[:] = qs[i]
                data.mocap_pos[:] = ms[i]
                mujoco.mj_forward(model, data)
                for angle in (45, 125, 225):
                    cam.azimuth = angle
                    renderer.update_scene(data, cam, scene_option=option)
                    path = args.output.parent / f"{args.output.stem}_step{i:03d}_az{angle}.png"
                    mediapy.write_image(str(path), renderer.render())
            print("Saved multiview snapshots", args.output.parent)
            return
        with mediapy.VideoWriter(str(args.output), shape=(720, 960), fps=20) as writer:
            for i, (q, m) in enumerate(zip(qs, ms)):
                data.qpos[:] = q
                data.mocap_pos[:] = m
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, cam, scene_option=option)
                frame = renderer.render()
                writer.add_image(frame)
                if i == min(600, len(qs) - 1):
                    mediapy.write_image(str(args.output.with_suffix(".png")), frame)
    print(args.output)


if __name__ == "__main__":
    main()
