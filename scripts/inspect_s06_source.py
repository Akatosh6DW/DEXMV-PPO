"""Audit and render the processed DexMV cylinder sample, not raw human RGB."""

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from prepare_cylinder_s02_source import load_archive, digest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/orca_cylinder_s02/source"
OUT = ROOT / "artifacts/orca_benchmark_suite/S06/source_audit"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    demos, member_hash = load_archive(SOURCE / "dexmv_demonstrations.zip")
    demo = demos["seq_96/multi_retargeting_v3.pkl"]
    model = mujoco.MjModel.from_xml_path(str(SOURCE / "processed/source_model.xml"))
    for field in ("body_pos", "body_quat", "site_pos", "site_quat"):
        getattr(model, field)[:] = demo["model_data"][0][field]
    data = mujoco.MjData(model)
    obj = model.body("tomato_soup_can_0").id
    center = np.load(SOURCE / "selection96/selected_segment.npz")["source_center"]
    prefixes = ("th", "ff", "mf", "rf", "lf")
    names = ("thumb", "index", "middle", "ring", "little")
    rows, tips = [], []
    frames = [182, 212, 242, 271]
    model.vis.global_.offwidth, model.vis.global_.offheight = 800, 600
    panels = []
    with mujoco.Renderer(model, 600, 800) as renderer:
        for frame in range(182, 272):
            state = demo["sim_data"][frame]
            data.qpos[:] = state["qpos"]
            data.qvel[:] = state["qvel"]
            mujoco.mj_forward(model, data)
            rot = data.xmat[obj].reshape(3, 3)
            points = np.array([data.site("S_" + p + "tip").xpos for p in prefixes])
            tips.append((points - data.xpos[obj]) @ rot - center)
            for c in data.contact[:data.ncon]:
                if obj not in model.geom_bodyid[c.geom] or c.dist > 0:
                    continue
                gid = c.geom2 if model.geom_bodyid[c.geom1] == obj else c.geom1
                name = model.geom(gid).name or model.body(model.geom_bodyid[gid]).name
                rows.append(dict(frame=frame, geom=name, depth_mm=-float(c.dist)*1000,
                                 **dict(zip(("x", "y", "z"), (c.pos-data.xpos[obj]) @ rot-center))))
            if frame in frames:
                for azimuth in (45, 225):
                    cam = mujoco.MjvCamera()
                    cam.lookat[:] = data.xpos[obj] + rot @ center
                    cam.distance, cam.elevation, cam.azimuth = .32, -20, azimuth
                    renderer.update_scene(data, cam)
                    pic = Image.fromarray(renderer.render())
                    ImageDraw.Draw(pic).text((12, 12), f"PROCESSED ADROIT, seq96 frame {frame}, view {azimuth}", fill="black")
                    pic.save(OUT / f"source_{frame}_{azimuth}.png")
                    panels.append(pic.resize((400, 300)))
    sheet = Image.new("RGB", (1600, 600))
    for i, p in enumerate(panels):
        sheet.paste(p, ((i//2)*400, (i%2)*300))
    sheet.save(OUT / "source_contact_sheet.jpg")
    with (OUT / "geometric_contacts.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "geom", "depth_mm", "x", "y", "z"])
        writer.writeheader()
        writer.writerows(rows)
    points = np.array(tips)
    report = dict(source="Published processed Adroit sequence96, frames182..271; raw RGB correspondence unverified",
                  limitation="Pose reconstruction shows proximity/intersection, not measured human contact or original dynamic forces",
                  archive_member_sha256=member_hash,
                  segment_sha256=digest(SOURCE / "selection96/selected_segment.npz"),
                  fingers=names, final_points_m=points[-1].tolist(),
                  last30_radial_distance_m=np.linalg.norm(points[-30:, :, :2], axis=-1).mean(0).tolist(),
                  touching_geometries={n: sum(r["geom"] == n for r in rows) for n in sorted({r["geom"] for r in rows})},
                  max_reconstructed_depth_mm=max((r["depth_mm"] for r in rows), default=0))
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
