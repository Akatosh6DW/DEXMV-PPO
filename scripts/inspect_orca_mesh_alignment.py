#!/usr/bin/env python3
"""Overlay original CAD visuals on saved proxy states without changing physics."""

import argparse
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from diagnose_orca_grasp import SCENE


def build_overlay(source, output):
    tree = ET.parse(source)
    root = tree.getroot()
    assets = root.find("asset")
    if assets is None:
        assets = ET.SubElement(root, "asset")
    compiler = root.find("compiler")
    meshdir = Path(compiler.get("meshdir", "")) if compiler is not None else Path()
    for mesh in assets.findall("mesh"):
        path = Path(mesh.get("file"))
        if not path.is_absolute():
            mesh.set("file", str((source.parent / meshdir / path).resolve()))
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    originals = SCENE.parent / "models/mjcf"
    for mesh in ET.parse(originals / "orcahand_right.mjcf").getroot().findall("asset/mesh"):
        visual = copy.deepcopy(mesh)
        visual.set("file", str((SCENE.parent / mesh.get("file")).resolve()))
        visual.set("scale", "0.001 0.001 0.001")
        assets.append(visual)
    bodies = {b.get("name"): b for b in root.iter("body")}
    added = []
    for body in ET.parse(originals / "orcahand_right_body.xml").getroot().iter("body"):
        if body.get("name") not in bodies:
            raise ValueError(f"Missing body: {body.get('name')}")
        for geom in body.findall("geom"):
            visual = copy.deepcopy(geom)
            material = visual.attrib.pop("material", "black")
            name = f"cad_visual_{len(added)}"
            visual.attrib.update(name=name, type="mesh", group="2", mass="0",
                                 contype="0", conaffinity="0",
                                 rgba="0.86 0.88 0.90 1" if material == "white" else "0.19 0.21 0.24 1")
            bodies[body.get("name")].append(visual)
            added.append(name)
    ET.indent(tree)
    tree.write(output, encoding="unicode")
    base = mujoco.MjModel.from_xml_path(str(source))
    overlay = mujoco.MjModel.from_xml_path(str(output))
    for key in ("nq", "nv", "nu", "nbody", "njnt"):
        if getattr(base, key) != getattr(overlay, key):
            raise AssertionError(f"Topology changed: {key}")
    for key in ("body_mass", "body_inertia", "body_pos", "body_quat", "jnt_pos",
                "jnt_axis", "jnt_range", "actuator_ctrlrange", "actuator_gainprm"):
        np.testing.assert_allclose(getattr(base, key), getattr(overlay, key), atol=1e-12)
    for name in added:
        geom = overlay.geom(name)
        assert geom.contype == 0 and geom.conaffinity == 0
    return overlay, dict(visual_geoms_added=len(added), physics_arrays_match=True,
                         source_model=str(source.resolve()), mode="render_only")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--trace", type=Path)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--build-only", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, report = build_overlay(args.model, args.output / "visual_overlay.xml")
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    if args.build_only:
        print(json.dumps(report))
        return
    if args.trace is None:
        p.error("--trace is required for rendering")
    import mediapy
    with np.load(args.trace) as z:
        qs, ms = z["qpos"][args.episode], z["mocap"][args.episode]
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [.065, .025, .245]
    cam.distance, cam.elevation = .36, -18
    option = mujoco.MjvOption()
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
    with mujoco.Renderer(model, 720, 960) as renderer:
        for i in (0, 20, 60, 600):
            data.qpos[:] = qs[i]
            data.mocap_pos[:] = ms[i]
            mujoco.mj_forward(model, data)
            for mode in ("proxy", "mesh", "overlay"):
                option.geomgroup[:] = 1
                option.geomgroup[2] = mode != "proxy"
                option.geomgroup[3] = mode != "mesh"
                for angle in (45, 225):
                    cam.azimuth = angle
                    renderer.update_scene(data, cam, scene_option=option)
                    mediapy.write_image(str(args.output / f"{mode}_step{i:03d}_az{angle}.png"), renderer.render())
    print(json.dumps(report))


if __name__ == "__main__":
    main()
