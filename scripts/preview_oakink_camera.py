"""Render camera diagnostics from one saved cap state; no physics stepping."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from inspect_orca_mesh_alignment import build_overlay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model, _ = build_overlay(args.model, args.output / "visual.xml")
    data = mujoco.MjData(model)
    with np.load(args.trace) as z:
        data.qpos[:] = z["qpos"][0, 60]
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.035, .025, .245]
    camera.distance, camera.elevation = .42, -18
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 480
    sheet = Image.new("RGB", (1920, 960))
    angles = (90, 135, 180, 225, 270, 315)
    with mujoco.Renderer(model, 480, 640) as renderer:
        for i, angle in enumerate(angles):
            camera.azimuth = angle
            renderer.update_scene(data, camera, scene_option=option)
            frame = Image.fromarray(renderer.render())
            ImageDraw.Draw(frame).text((12, 12), f"azimuth={angle}; elevation=-18; t=3.05s", fill="white")
            frame.save(args.output / f"az{angle}.png")
            sheet.paste(frame, ((i % 3) * 640, (i // 3) * 480))
    sheet.save(args.output / "cameras.png")
    (args.output / "report.json").write_text(json.dumps(dict(angles=angles, no_resimulation=True, frame=60, episode=0)))


if __name__ == "__main__":
    main()
