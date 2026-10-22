"""Render native development states; never re-simulate them for presentation."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from inspect_orca_mesh_alignment import build_overlay


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    m, _ = build_overlay(a.input / "model.xml", a.output / "overlay.xml")
    z = np.load(a.trace)
    d = mujoco.MjData(m)
    ref = np.load(a.input / "reference.npz")
    cam = mujoco.MjvCamera()
    cam.lookat[:] = ref["object_position"]
    cam.distance, cam.elevation = .30, -20
    m.vis.global_.offwidth, m.vis.global_.offheight = 800, 600
    panels = []
    with mujoco.Renderer(m, 600, 800) as r:
        for frame in (0, min(59, len(z["qpos"])-1), len(z["qpos"])-1):
            d.qpos[:], d.qvel[:], d.mocap_pos[:] = z["qpos"][frame], z["qvel"][frame], z["mocap"][frame]
            mujoco.mj_forward(m, d)
            for az in (45, 225):
                cam.azimuth = az
                opt = mujoco.MjvOption()
                opt.geomgroup[3] = 0
                r.update_scene(d, cam, scene_option=opt)
                pic = Image.fromarray(r.render())
                ImageDraw.Draw(pic).text((10, 10), f"S06 reference probe t={(frame+1)*.05:.2f}s az={az}", fill="white")
                pic.save(a.output / f"frame_{frame}_az{az}.png")
                panels.append(pic.resize((400, 300)))
    sheet = Image.new("RGB", (1200, 600))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i//2)*400, (i%2)*300))
    sheet.save(a.output / "sheet.jpg")


if __name__ == "__main__":
    main()
