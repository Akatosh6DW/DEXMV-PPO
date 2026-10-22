# 中文阅读副本：02 绑定同源物体及坐标
# 原始入口：scripts/prepare_hic_orca_object.py
# 仅用于阅读，保留原路径假设；不要从此副本启动训练。运行请使用原始入口。
# 本副本只增加注释，生成时核验AST与非注释词法序列完全一致。
"""Prepare a source-sized HIC object and synchronized, unscaled hand/object data.

This is a geometry/data preparation stage, not a validated Orca controller.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/cube20/hic20"
MODEL = ROOT / "artifacts/orca_collision_refit/G08_thumb_convex_candidate/cube.xml"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# 导读：把每帧手指位置转到该帧物体坐标系；移动相机不应改变抓在物体哪个面上。
def object_local(points, poses):
    points, poses = np.asarray(points), np.asarray(poses)
    if points.ndim != 3 or poses.shape != (len(points), 4, 4):
        raise ValueError("Expected synchronized [T,N,3] points and [T,4,4] object poses")
    if points.shape[-1] != 3 or not np.isfinite(points).all() or not np.isfinite(poses).all():
        raise ValueError("Invalid points or poses")
    rotations = poses[:, :3, :3]
    if (not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4)
            or not np.allclose(poses[:, 3], [0, 0, 0, 1])):
        raise ValueError("Object poses must be proper rigid transforms")
    return np.einsum("tji,tkj->tki", rotations, points - poses[:, None, :3, 3])


def compiled_vertices(model, geom_name="cube"):
    geom = model.geom(geom_name)
    if int(geom.type[0]) != mujoco.mjtGeom.mjGEOM_MESH:
        raise ValueError("Expected source mesh, not a substitute primitive")
    mid = int(geom.dataid[0])
    start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, geom.quat)
    return model.mesh_vert[start:start + count] @ rot.reshape(3, 3).T + geom.pos


# 导读：绑定同一HIC序列的网格、跟踪和重定向产物，检查单位与尺寸。
def build(source, model_path, output):
    off = source / "models_motion/20/Model_Object_Cube.OFF"
    human_path = source / "prepared/human_cube.npz"
    dexmv_path = source / "dexmv_retarget/adroit_cube.npz"
    video_path = source / "videos/20/input_rgb.mp4"
    mesh = trimesh.load(off, force="mesh", process=True)
    mesh.apply_scale(.001)
    if not mesh.is_watertight or not mesh.is_convex:
        raise ValueError("This converter requires the convex, closed HIC20 object")
    if not np.allclose(mesh.bounds.mean(0), 0, atol=1e-7):
        raise ValueError("Object origin convention changed; do not silently recenter")
    with np.load(human_path) as z:
        human = {k: z[k] for k in ("hand_joints", "object_poses", "time", "source_video_frame")}
    with np.load(dexmv_path) as z:
        dex = {k: z[k] for k in ("fitted_positions", "target_positions", "object_poses", "time", "camera_to_adroit")}
    np.testing.assert_allclose(human["time"], dex["time"], atol=1e-7)
    np.testing.assert_allclose(dex["object_poses"], dex["camera_to_adroit"] @ human["object_poses"], atol=1e-6)
    video = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json", str(video_path)
    ]))["streams"][0]
    fps_num, fps_den = map(int, video["r_frame_rate"].split("/"))
    if int(video["nb_frames"]) != len(human["time"]):
        raise ValueError("Video/tracking frame counts differ")
    np.testing.assert_allclose(np.diff(human["time"]), fps_den / fps_num, atol=1e-7)

    tree = ET.parse(model_path)
    root = tree.getroot()
    # Resolve existing hand assets before moving the generated XML.
    compiler = root.find("compiler")
    meshdir = Path(compiler.get("meshdir", "")) if compiler is not None else Path()
    for asset in root.findall("asset/mesh"):
        file = Path(asset.get("file"))
        if not file.is_absolute():
            asset.set("file", str((model_path.parent / meshdir / file).resolve()))
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    cube = root.find(".//geom[@name='cube']")
    if cube is None or cube.get("type") != "box":
        raise ValueError("Expected the existing box task as input")
    old_size = 2 * np.fromstring(cube.get("size"), sep=" ")
    old_model = mujoco.MjModel.from_xml_path(str(model_path))
    mass = float(old_model.body("cube").mass[0])
    output.mkdir(parents=True, exist_ok=False)
    asset_path = output / "hic20_cube_m.stl"
    mesh.export(asset_path)
    ET.SubElement(root.find("asset"), "mesh", name="hic20_cube_source",
                  file=asset_path.name, maxhullvert="64")
    cube.set("type", "mesh")
    cube.set("mesh", "hic20_cube_source")
    cube.attrib.pop("size")
    # Preserve assumed mass, but let MuJoCo recompute inertia for the new geometry.
    root.find(".//body[@name='cube']").set("pos", ".055 .025 .265")
    support_z = .265 + float(mesh.bounds[0, 2]) - .004
    root.find(".//body[@name='cube_support']").set("pos", f".055 .025 {support_z:.10g}")
    tree.write(output / "cube.xml", encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(output / "cube.xml"))
    compiled = compiled_vertices(model)
    np.testing.assert_allclose(np.ptp(compiled, axis=0), mesh.extents, atol=1e-7)
    np.testing.assert_allclose(model.body("cube").mass, mass, atol=1e-10)
    for body in range(old_model.nbody):
        if body == old_model.body("cube").id:
            continue
        np.testing.assert_allclose(old_model.body_mass[body], model.body_mass[body])
        np.testing.assert_allclose(old_model.body_inertia[body], model.body_inertia[body])
    for field in ("jnt_range", "actuator_ctrlrange", "actuator_gainprm", "actuator_biasprm"):
        np.testing.assert_array_equal(getattr(old_model, field), getattr(model, field))

    local_human = object_local(human["hand_joints"], human["object_poses"])
    local_fitted = object_local(dex["fitted_positions"], dex["object_poses"])
    local_targets = object_local(dex["target_positions"], dex["object_poses"])
    data_path = output / "source_correspondence.npz"
    np.savez_compressed(data_path, time=human["time"], source_video_frame=human["source_video_frame"],
                        object_poses_camera=human["object_poses"],
                        object_poses_dexmv=dex["object_poses"],
                        human_joints_object_local=local_human,
                        dexmv_fitted_object_local=local_fitted,
                        dexmv_targets_object_local=local_targets,
                        object_bounds_m=mesh.bounds, spatial_scale=np.array(1.0))
    sources = (off, human_path, dexmv_path, video_path, model_path)
    report = dict(
        stage="source_object_and_correspondence_prepared_not_controller_validated",
        trainable=False, source_sequence=20, source_object="Rubik-style cube (single rigid body)",
        source_mesh_units="millimeters", simulation_units="meters", spatial_scale=1.,
        source_extents_m=mesh.extents.tolist(), compiled_extents_m=np.ptp(compiled, axis=0).tolist(),
        old_task_extents_m=old_size.tolist(), old_to_source_scale=(old_size / mesh.extents).tolist(),
        source_vertices=len(mesh.vertices), source_faces=len(mesh.faces),
        convex_hull_vertices=len(mesh.convex_hull.vertices),
        source_volume_m3=float(mesh.volume), convex_hull_volume_m3=float(mesh.convex_hull.volume),
        assumed_mass_kg=mass, assumed_friction=model.geom("cube").friction.tolist(),
        mass_friction_status="inherited simulation assumptions, not measured from video",
        cube_inertia_kg_m2=model.body("cube").inertia.tolist(), video=video,
        source_object_translation_extent_m=np.ptp(human["object_poses"][:, :3, 3], axis=0).tolist(),
        object_pose_convention="M(t) @ inverse(M(0)); bind frame omitted; no independent hand/object alignment",
        old_reference_compatible=False,
        pending=["Orca reachability and object-relative contact retargeting",
                 "Reset pose/support alignment and collision validation",
                 "Native and MJX dynamics gates before training"],
        task_scope="Grasp/hold transfer only; fixed-wrist task does not reproduce video transport",
        visual_scope="Source shape retained; colored stickers/internal rotating layers not modeled",
        sha256={str(p.resolve()): digest(p) for p in sources},
        generated_sha256={p.name: digest(p) for p in (output / "cube.xml", asset_path, data_path)},
    )
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
        "-vf", "select='eq(n,0)+eq(n,45)+eq(n,95)+eq(n,140)',scale=640:-1,tile=2x2",
        "-frames:v", "1", str(output / "source_frames_0_45_95_140.png")
    ], check=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = build(args.source, args.model, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
