"""Rigidly reorient a matched hand/object pair without re-fitting its fingers."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from diagnose_orca_grasp import model_indices
from prepare_hic_orca_object import compiled_vertices, digest


def quaternion(matrix):
    return Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]


def matrix(quat):
    return Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()


def verify_transform(model_path, reference_path, manifest_path):
    proof = json.loads(Path(manifest_path).read_text())
    parent_model = Path(proof["parent_model"])
    parent_reference = Path(proof["parent_reference"])
    for path, expected in ((parent_model, proof["parent_model_sha256"]),
                           (parent_reference, proof["parent_reference_sha256"]),
                           (model_path, proof["model_sha256"]),
                           (reference_path, proof["reference_sha256"])):
        if digest(path) != expected:
            raise ValueError("Rigid-transform provenance changed: " + str(path))
    rotation, translation = np.array(proof["rotation"]), np.array(proof["translation"])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1., atol=1e-12)
    old = mujoco.MjModel.from_xml_path(str(parent_model))
    new = mujoco.MjModel.from_xml_path(str(model_path))
    with np.load(parent_reference) as z:
        before = {k: z[k] for k in z.files}
    with np.load(reference_path) as z:
        after = {k: z[k] for k in z.files}
    for key in ("ctrl", "raw_ik_ctrl", "initial_ctrl", "time", "source_time",
                "source_local_tips", "spatial_scale", "object_extents_m", "source_frame_start"):
        np.testing.assert_array_equal(before[key], after[key])
    np.testing.assert_allclose(after["target_tips"], before["target_tips"] @ rotation.T + translation, atol=1e-12)
    np.testing.assert_allclose(after["object_position"], rotation @ before["object_position"] + translation, atol=1e-12)
    np.testing.assert_allclose(after["object_to_world_rotation"], rotation @ before["object_to_world_rotation"], atol=1e-12)
    np.testing.assert_allclose(matrix(after["object_quaternion"]), after["object_to_world_rotation"], atol=1e-12)
    for field in ("body_mass", "body_inertia", "jnt_range", "jnt_axis",
                  "actuator_ctrlrange", "actuator_gainprm", "actuator_biasprm", "geom_size", "geom_pos",
                  "geom_quat", "geom_type", "geom_friction", "geom_contype", "geom_conaffinity"):
        np.testing.assert_array_equal(getattr(new, field), getattr(old, field))
    for i in range(old.nbody):
        name = old.body(i).name
        if name == "right_mount":
            assert old.body_mass[i] == new.body_mass[i] == 0.
            np.testing.assert_allclose(new.body_pos[i], rotation @ old.body_pos[i] + translation, atol=1e-12)
            np.testing.assert_allclose(matrix(new.body_quat[i]), rotation @ matrix(old.body_quat[i]), atol=1e-12)
            np.testing.assert_allclose(new.body_ipos[i], rotation @ old.body_ipos[i] + translation, atol=1e-12)
            np.testing.assert_allclose(matrix(new.body_iquat[i]), rotation @ matrix(old.body_iquat[i]), atol=1e-12)
        else:
            np.testing.assert_array_equal(new.body_ipos[i], old.body_ipos[i])
            np.testing.assert_array_equal(new.body_iquat[i], old.body_iquat[i])
        if name not in ("right_mount", "cube", "cube_support"):
            np.testing.assert_array_equal(new.body_pos[i], old.body_pos[i])
            np.testing.assert_array_equal(new.body_quat[i], old.body_quat[i])
    qids, cube_qid, _ = model_indices(old)
    old_data, new_data = mujoco.MjData(old), mujoco.MjData(new)
    bodies = [i for i in range(old.nbody) if old.body(i).name not in ("world", "cube_support")]
    maximum = 0.
    for ctrl in before["raw_ik_ctrl"][::5]:
        for data, model, packet in ((old_data, old, before), (new_data, new, after)):
            data.qpos[qids] = ctrl
            data.qpos[cube_qid:cube_qid + 7] = np.r_[packet["object_position"], packet["object_quaternion"]]
            mujoco.mj_kinematics(model, data)
        expected = old_data.xpos[bodies] @ rotation.T + translation
        maximum = max(maximum, float(np.abs(expected - new_data.xpos[bodies]).max()))
        np.testing.assert_allclose(new_data.xpos[bodies], expected, atol=1e-12)
        expected_rot = rotation @ old_data.xmat[bodies].reshape(-1, 3, 3)
        np.testing.assert_allclose(new_data.xmat[bodies].reshape(-1, 3, 3), expected_rot, atol=1e-12)
    return dict(max_body_position_error_m=maximum, checked_frames=len(before["raw_ik_ctrl"][::5]),
                common_rigid_transform_pass=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--angle", type=float, default=180.)
    args = p.parse_args()
    if not np.isfinite(args.angle):
        p.error("Nonfinite rotation")
    old = mujoco.MjModel.from_xml_path(str(args.model))
    with np.load(args.reference) as z:
        packet = {k: z[k] for k in z.files}
    rotation = Rotation.from_euler("x", args.angle, degrees=True).as_matrix()
    center = packet["object_position"].copy()
    translation = center - rotation @ center
    packet["object_to_world_rotation"] = rotation @ packet["object_to_world_rotation"]
    packet["object_quaternion"] = quaternion(packet["object_to_world_rotation"])
    packet["target_tips"] = packet["target_tips"] @ rotation.T + translation
    tree = ET.parse(args.model)
    for mesh in tree.findall("asset/mesh"):
        path = Path(mesh.get("file"))
        if not path.is_absolute():
            mesh.set("file", str((args.model.parent / path).resolve()))
    root = tree.find(".//body[@name='right_mount']")
    mount = old.body("right_mount")
    root.set("pos", " ".join(map(str, rotation @ mount.pos + translation)))
    root.set("quat", " ".join(map(str, quaternion(rotation @ matrix(mount.quat)))))
    tree.find(".//body[@name='cube']").set("quat", " ".join(map(str, packet["object_quaternion"])))
    lowest = np.min((compiled_vertices(old) @ packet["object_to_world_rotation"].T)[:, 2])
    tree.find(".//body[@name='cube_support']").set("pos", " ".join(map(str, center + [0, 0, lowest - .004])))
    args.output.mkdir(parents=True, exist_ok=False)
    new_model, new_ref = args.output / "cube.xml", args.output / "reference.npz"
    tree.write(new_model, encoding="unicode")
    np.savez_compressed(new_ref, **packet)
    proof = dict(parent_model=str(args.model.resolve()), parent_reference=str(args.reference.resolve()),
                 parent_model_sha256=digest(args.model), parent_reference_sha256=digest(args.reference),
                 model_sha256=digest(new_model), reference_sha256=digest(new_ref),
                 rotation=rotation.tolist(), translation=translation.tolist(), pivot=center.tolist(),
                 angle_deg=args.angle, axis="world_x", gravity_rotated=False,
                 note="Hand mounting and object rotated together; support stays below; controls/local trajectories unchanged")
    proof_path = args.output / "mount_transform.json"
    proof_path.write_text(json.dumps(proof, indent=2) + "\n")
    checks = verify_transform(new_model, new_ref, proof_path)
    proxy = json.loads((args.model.parent / "object_proxy_report.json").read_text())
    proxy.update(parent_model_sha256=digest(args.model), model_sha256=digest(new_model))
    (args.output / "object_proxy_report.json").write_text(json.dumps(proxy, indent=2) + "\n")
    (args.output / "transform_source.py").write_bytes(Path(__file__).read_bytes())
    from orca_residual_controls import load_reference
    from probe_hic53_orca import native_rollout
    model = mujoco.MjModel.from_xml_path(str(new_model))
    base, _ = load_reference(new_ref, *model.actuator_ctrlrange.T, packet["initial_ctrl"], .05, 800, .1)
    native = []
    for seed in range(3):
        trace, check = native_rollout(model, center, packet["object_quaternion"], base, seed,
                                      initial_ctrl=packet["initial_ctrl"])
        np.savez_compressed(args.output / f"native_seed{seed}.npz", **trace)
        native.append(check)
        print("NATIVE", json.dumps(check), flush=True)
    report = dict(proof, kinematic_checks=checks, native_checks=native, trainable=False,
                  endpoint_target_height_mm=((packet["target_tips"][-1] - center)[:, 2] * 1000).tolist())
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(checks), flush=True)


if __name__ == "__main__":
    main()
