"""Render and audit prescribed landmark IK against source-sized object meshes.

The cap is a mocap body in this diagnostic. Its rotation is prescribed by the
source annotation, not caused by the robot. These traces are never RL results.
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
import trimesh

from prepare_hic_orca_object import digest, compiled_vertices


def vec(value):
    return " ".join(f"{x:.12g}" for x in np.asarray(value).ravel())


def build_scene(model_path, mesh_source, transform, cap_pose, output):
    tree = ET.parse(model_path)
    root = tree.getroot()
    root.set("model", "Orca OakInk prescribed kinematic audit NOT dynamics")
    world, assets = root.find("worldbody"), root.find("asset")
    for name in ("cube", "cube_support"):
        world.remove(world.find(f"body[@name='{name}']"))
    for geom in root.findall(".//geom"):
        if geom.get("contype") == "1":
            geom.set("conaffinity", "6")
    reports = {}
    for name in ("bottle", "cap"):
        mesh = trimesh.load(mesh_source / f"{name}.ply", force="mesh", process=True)
        mesh.export(output / f"{name}_source.stl")
        mesh.convex_hull.export(output / f"{name}_convex.stl")
        for kind in ("source", "convex"):
            ET.SubElement(assets, "mesh", name=f"oakink_{name}_{kind}", file=f"{name}_{kind}.stl")
        pose = transform if name == "bottle" else cap_pose
        attrs = dict(name=f"oakink_{name}", pos=vec(pose[:3, 3]),
                     quat=vec(Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]))
        if name == "cap":
            attrs["mocap"] = "true"
        body = ET.SubElement(world, "body", **attrs)
        ET.SubElement(body, "geom", name=f"{name}_visual", type="mesh", mesh=f"oakink_{name}_source",
                      contype="0", conaffinity="0", group="1", mass="0",
                      rgba=".85 .85 .88 1" if name == "cap" else ".25 .62 .58 1")
        ET.SubElement(body, "geom", name=f"{name}_collision", type="mesh", mesh=f"oakink_{name}_convex",
                      contype="2" if name == "cap" else "4", conaffinity="1", group="3",
                      mass=".015" if name == "cap" else ".1", rgba="1 .3 .1 .3")
        reports[name] = dict(source_sha256=digest(mesh_source / f"{name}.ply"), source_bounds_m=mesh.bounds.tolist(),
                             source_watertight=bool(mesh.is_watertight), hull_bounds_m=mesh.convex_hull.bounds.tolist(),
                             proxy="unscaled whole-object convex hull; neck concavity and cap interior filled",
                             compiled_hull_surface_accuracy_validated=False)
    for i in range(5):
        body = ET.SubElement(world, "body", name=f"target_{i}", mocap="true")
        ET.SubElement(body, "geom", name=f"target_marker_{i}", type="sphere", size=".002", mass="0",
                      contype="0", conaffinity="0", group="4", rgba="1 .1 .1 1")
    for asset in assets.findall("mesh"):
        path = Path(asset.get("file"))
        if not path.is_absolute() and not asset.get("name").startswith("oakink_"):
            asset.set("file", str((model_path.parent / path).resolve()))
    ET.indent(tree)
    tree.write(output / "kinematic_scene.xml", encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(output / "kinematic_scene.xml"))
    for name in ("cap", "bottle"):
        np.testing.assert_allclose(np.ptp(compiled_vertices(model, f"{name}_collision"), axis=0),
                                   np.ptp(reports[name]["source_bounds_m"], axis=0), atol=1e-7)
    return model, reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/orca_collision_refit/O20_hic53_elliptic/cube.xml"))
    parser.add_argument("--meshes", type=Path, default=Path("artifacts/orca_formal/CAP01_source_inspection"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.reference) as source:
        z = {k: source[k] for k in source.files}
    args.output.mkdir(parents=True, exist_ok=False)
    model, mesh_report = build_scene(args.model, args.meshes, z["bottle_to_simulation"],
                                    z["cap_to_simulation"][0], args.output)
    data = mujoco.MjData(model)
    qids = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    cap_mocap = model.body("oakink_cap").mocapid[0]
    target_mocaps = [model.body(f"target_{i}").mocapid[0] for i in range(5)]
    rows, contact_rows = [], []
    for i, ctrl in enumerate(z["ctrl"]):
        data.qpos[qids] = ctrl
        pose = z["cap_to_simulation"][i]
        data.mocap_pos[cap_mocap] = pose[:3, 3]
        data.mocap_quat[cap_mocap] = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        data.mocap_pos[target_mocaps] = z["target_tips"][i]
        mujoco.mj_forward(model, data)
        depths, contact_names = {}, []
        for name in ("cap", "bottle"):
            object_geom = model.geom(f"{name}_collision").id
            depth = 0.
            for contact in data.contact[:data.ncon]:
                if object_geom not in (contact.geom1, contact.geom2):
                    continue
                other = contact.geom2 if contact.geom1 == object_geom else contact.geom1
                if model.geom_contype[other] != 1:
                    continue
                depth = max(depth, -float(contact.dist))
                if contact.dist < 0:
                    contact_names.append(dict(object=name, hand_geom=model.geom(other).name, depth_m=-float(contact.dist)))
            depths[f"{name}_penetration_m"] = depth
        rows.append(dict(qpos=data.qpos.copy(), mocap=data.mocap_pos.copy(), mocap_quat=data.mocap_quat.copy(), **depths))
        contact_rows.append(contact_names)
    trace = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(args.output / "prescribed_trace.npz", **trace, source_frame_id=z["source_frame_id"])
    phase = (z["source_frame_id"] >= 423) & (z["source_frame_id"] < 1548)
    report = dict(kind="prescribed_kinematic_collision_audit_not_RL_or_physical_cap_rotation",
                  reference_sha256=digest(args.reference), model_sha256=digest(args.output / "kinematic_scene.xml"),
                  source_sha256=digest(Path(__file__)), mesh_report=mesh_report,
                  phase="annotated unscrew before cap lift/transport; source IDs [423,1548)",
                  cap_max_depth_mm=float(1000 * trace["cap_penetration_m"][phase].max()),
                  bottle_max_depth_mm=float(1000 * trace["bottle_penetration_m"][phase].max()),
                  cap_over_3mm_fraction=float((trace["cap_penetration_m"][phase] > .003).mean()),
                  bottle_over_3mm_fraction=float((trace["bottle_penetration_m"][phase] > .003).mean()),
                  no_dynamic_success_claim=True, cap_bottle_collision_disabled=True,
                  per_frame_contacts=contact_rows)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    model.vis.global_.offwidth, model.vis.global_.offheight = 800, 640
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.075, .035, .25]
    camera.distance, camera.elevation = .40, -15
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 1, 1, 0, 0]
    # This source XML contains collision proxies but no full visual hand mesh.
    # Keep those proxies visible and label the images as collision diagnostics.
    sheet = Image.new("RGB", (1600, 1280), "white")
    with mujoco.Renderer(model, 640, 800) as renderer:
        for index, frame in enumerate((423, 1099, 1448, 1548)):
            i = int(np.argmin(abs(z["source_frame_id"] - frame)))
            data.qpos[:] = trace["qpos"][i]
            data.mocap_pos[:] = trace["mocap"][i]
            data.mocap_quat[:] = trace["mocap_quat"][i]
            mujoco.mj_forward(model, data)
            camera.azimuth = 125
            renderer.update_scene(data, camera, scene_option=option)
            image = Image.fromarray(renderer.render())
            ImageDraw.Draw(image).text((12, 12), f"PRESCRIBED IK / NOT RL  source frame {z['source_frame_id'][i]}",
                                      fill="white", stroke_width=1, stroke_fill="black")
            image.save(args.output / f"frame_{int(z['source_frame_id'][i])}.png")
            sheet.paste(image, ((index % 2) * 800, (index // 2) * 640))
    sheet.save(args.output / "kinematic_collision_preview.png")
    print(json.dumps({k: v for k, v in report.items() if k not in ("mesh_report", "per_frame_contacts")}), flush=True)


if __name__ == "__main__":
    main()
