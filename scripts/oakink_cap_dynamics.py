"""Source-sized passive screw cap with state-dependent thread disengagement.

This development model is not yet a frozen experimental task. The demonstrator
poses determine geometry/reference only, never the simulated cap's state.
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import trimesh

from prepare_hic_orca_object import digest


def vec(values):
    return " ".join(f"{x:.12g}" for x in np.asarray(values).ravel())


def motion_parameters(reference):
    ids, poses = reference["source_frame_id"], reference["source_cap_in_bottle"]
    anchor = int(np.argmin(abs(ids - 423)))
    angles = np.unwrap(np.arctan2(poses[:, 1, 0], poses[:, 0, 0]))
    angles -= angles[anchor]
    centers = np.einsum("tij,j->ti", poses[:, :3, :3], [.000173, .000358, .050284]) + poses[:, :3, 3]
    mask = (ids >= 900) & (ids < 1490)
    result = least_squares(lambda x: centers[mask, 2] - (x[0] + x[1] * angles[mask]),
                           [.05, .0005], loss="soft_l1", f_scale=.001)
    pitch = float(result.x[1])
    if not 0 < pitch < .002:
        raise ValueError("Unexpected video-fit pitch; inspect source motion instead of clamping it")
    initial_z = float(np.median(centers[(ids >= 423) & (ids < 875), 2]))
    lifted = np.flatnonzero((ids > 1300) & (centers[:, 2] > initial_z + .010))
    if not len(lifted):
        raise ValueError("No observed cap separation segment")
    release = float(np.max(angles[anchor:lifted[0] + 1]))
    return dict(pitch_m_per_rad=pitch, pitch_mm_per_turn=pitch * 2 * np.pi * 1000,
                release_angle_rad=release, release_angle_degrees=float(np.rad2deg(release)),
                separation_source_frame=int(ids[lifted[0]]),
                fitted_z_residual_p95_mm=float(1000 * np.percentile(abs(result.fun), 95)),
                status="Approximate helical motion fitted to video-aligned object poses, not a measured thread specification")


def build(scene, collision, reference_path, output, axis_aligned=False, installation_lift=0., bottle_collision=None,
          thread_contacts="mesh"):
    with np.load(reference_path) as packet:
        reference = dict(packet)
    output.mkdir(parents=True, exist_ok=False)
    (output / "dynamics_source.py").write_bytes(Path(__file__).read_bytes())
    parameters = motion_parameters(reference)
    tree = ET.parse(scene)
    root = tree.getroot()
    root.set("model", "OakInk fixed bottle passive unthreading development")
    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "numeric", name="analytic_thread_contacts", data=str(int(thread_contacts == "analytic")))
    root.find("option").set("timestep", ".002")
    root.find("option").set("iterations", "60")
    world, assets = root.find("worldbody"), root.find("asset")
    for body in list(world.findall("body")):
        if body.get("name", "").startswith("target_"):
            world.remove(body)
    bottle, cap = world.find("body[@name='oakink_bottle']"), world.find("body[@name='oakink_cap']")
    world.remove(cap)
    bottle.append(cap)
    bottle.attrib.pop("mocap", None)
    cap.attrib.pop("mocap", None)
    anchor = int(np.argmin(abs(reference["source_frame_id"] - 423)))
    source_installed = reference["source_cap_in_bottle"][anchor]
    installed = source_installed.copy()
    if axis_aligned:
        yaw = np.arctan2(installed[1, 0], installed[0, 0])
        installed[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
        installed[:2, 3] = 0.
    installed[2, 3] += installation_lift
    cap.set("pos", vec(installed[:3, 3]))
    cap.set("quat", "1 0 0 0")
    installed_quat = Rotation.from_matrix(installed[:3, :3]).as_quat()[[3, 0, 1, 2]]
    for geom in cap.findall("geom"):
        geom.set("quat", vec(installed_quat))
    for name, body in (("cap", cap), ("bottle", bottle)):
        body.remove(body.find(f"geom[@name='{name}_collision']"))
        asset_root = bottle_collision if name == "bottle" and bottle_collision is not None else collision
        files = sorted((asset_root / name).glob("part_*.stl"))
        if not files:
            raise ValueError("Missing decomposed source object")
        for i, path in enumerate(files):
            mesh_name = f"{name}_decomposed_{i:02d}"
            ET.SubElement(assets, "mesh", name=mesh_name, file=str(path.resolve()))
            attributes = dict(name=mesh_name, type="mesh", mesh=mesh_name, group="3",
                              contype="2" if name == "cap" else "4", conaffinity="5" if name == "cap" else "3",
                              mass="0", friction="1.35 .025 .004", rgba="1 .4 .15 .3")
            if name == "cap":
                attributes["quat"] = vec(installed_quat)
            ET.SubElement(body, "geom", **attributes)
    # Preserve a documented mass/inertia estimate independently of proxy pieces.
    hull = trimesh.load(collision / "cap_source.stl", force="mesh", process=True).convex_hull
    hull.density = .015 / hull.volume
    tensor = installed[:3, :3] @ hull.moment_inertia @ installed[:3, :3].T
    eig, axes = np.linalg.eigh(tensor)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    ET.SubElement(cap, "inertial", pos=vec(installed[:3, :3] @ hull.center_mass), mass=".015",
                  diaginertia=vec(eig), quat=vec(Rotation.from_matrix(axes).as_quat()[[3, 0, 1, 2]]))
    for name, kind, axis in (("cap_x", "slide", "1 0 0"), ("cap_y", "slide", "0 1 0"),
                             ("cap_lift", "slide", "0 0 1"), ("cap_turn", "hinge", "0 0 1"),
                             ("cap_tilt_x", "hinge", "1 0 0"), ("cap_tilt_y", "hinge", "0 1 0")):
        joint = ET.SubElement(cap, "joint", name=name, type=kind, axis=axis, limited="false",
                      damping=".03" if kind == "slide" else ".0002",
                      armature=".00001", frictionloss=".001" if name == "cap_turn" else "0")
        if name == "cap_turn":
            joint.set("limited", "true")
            joint.set("range", "-.05 1000")
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    for name in ("cap_x", "cap_y", "cap_tilt_x", "cap_tilt_y"):
        ET.SubElement(equality, "joint", name="thread_" + name, joint1=name, polycoef="0 0 0 0 0",
                      solref=".004 1", solimp=".999 .9999 .001")
    ET.SubElement(equality, "joint", name="thread_pitch", joint1="cap_lift", joint2="cap_turn",
                  polycoef=vec([0, parameters["pitch_m_per_rad"], 0, 0, 0]),
                  solref=".004 1", solimp=".999 .9999 .001")
    # The cap is a child of the static bottle: explicitly allow their contact.
    root.find("option").append(ET.Element("flag", filterparent="disable"))
    for mesh in assets.findall("mesh"):
        path = Path(mesh.get("file"))
        if not path.is_absolute():
            mesh.set("file", str((scene.parent / path).resolve()))
    ET.indent(tree)
    tree.write(output / "model.xml", encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(output / "model.xml"))
    assert model.nu == 17 and model.nmocap == 0
    qids = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    data = mujoco.MjData(model)
    data.qpos[qids], data.ctrl[:] = reference["ctrl"][anchor], reference["ctrl"][anchor]
    mujoco.mj_forward(model, data)
    geometric_initial_contacts = contact_report(model, data)
    apply_thread_contact_mode(model, data)
    mujoco.mj_forward(model, data)
    initial_contacts = contact_report(model, data)
    report = dict(stage="development_passive_thread_not_validated", parameters=parameters,
                  model_sha256=digest(output / "model.xml"), source_scene_sha256=digest(scene),
                  collision_report_sha256=digest(collision / "report.json"), reference_sha256=digest(reference_path),
                  bottle_collision_report_sha256=digest((bottle_collision or collision) / "report.json"),
                  initial_contacts=initial_contacts, initial_ctrl=reference["ctrl"][anchor].tolist(),
                  geometric_initial_contacts_before_thread_filter=geometric_initial_contacts,
                  source_installed_pose=source_installed.tolist(), simulated_installed_pose=installed.tolist(),
                  axis_aligned_installation=axis_aligned, installation_lift_m=installation_lift,
                  source_anchor_frame=int(reference["source_frame_id"][anchor]),
                  mass_kg=.015, mass_and_inertia_measured=False, cap_actuators=0,
                  thread_contact_model=thread_contacts,
                  tightening_stop_rad=-.05,
                  cap_bottle_contact_enabled="after thread disengagement" if thread_contacts == "analytic" else "always",
                  thread_release="All five constraints disengage at observed separation angle; never reactivate",
                  limitations=["Source pose fitting can leave initial mesh overlap; must pass geometry gate",
                               "Linear pitch and dry friction approximate unmeasured screw mechanics",
                               "Analytic mode substitutes joint constraints for thread contact while engaged; mesh overlap is separately reported, not claimed absent",
                               "Six scalar passive joints approximate a free cap after unthreading",
                               "Source hand arm transport is not reproduced by a fixed-base hand"],
                  documentation="https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality-joint")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return model, report


def contact_report(model, data):
    summary = dict(cap_bottle_depth_m=0., hand_cap_depth_m=0., hand_bottle_depth_m=0.)
    for c in data.contact[:data.ncon]:
        types = sorted([int(model.geom_contype[c.geom1]), int(model.geom_contype[c.geom2])])
        name = {(2, 4): "cap_bottle_depth_m", (1, 2): "hand_cap_depth_m", (1, 4): "hand_bottle_depth_m"}.get(tuple(types))
        if name:
            summary[name] = max(summary[name], -float(c.dist))
    return summary


def apply_thread_contact_mode(model, data):
    engaged = bool(data.eq_active.any())
    model.jnt_limited[model.joint("cap_turn").id] = int(engaged)
    flag = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "analytic_thread_contacts")
    if flag < 0 or model.numeric_data[model.numeric_adr[flag]] == 0:
        return
    # Recompute from restored per-episode equality state on every native call.
    # Shared model scratch masks are not hidden episode state; callback users
    # must hold their model lock across this function, stepping and packing.
    cap = (model.geom_bodyid == model.body("oakink_cap").id) & (model.geom_contype == 2)
    bottle = (model.geom_bodyid == model.body("oakink_bottle").id) & (model.geom_contype == 4)
    model.geom_conaffinity[cap] = 1 if engaged else 5
    model.geom_conaffinity[bottle] = 1 if engaged else 3


def step_passive(model, data, release_angle, substeps=25):
    turn = model.joint("cap_turn").qposadr[0]
    for _ in range(substeps):
        if data.qpos[turn] >= release_angle:
            data.eq_active[:] = False
        apply_thread_contact_mode(model, data)
        mujoco.mj_step(model, data)
        bad = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
               mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL)
        if any(data.warning[int(k)].number for k in bad) or not np.isfinite(data.qpos).all():
            raise FloatingPointError("Invalid cap physics; not a valid failed episode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("artifacts/orca_formal/CAP11_sequence_mount_audit/kinematic_scene.xml"))
    parser.add_argument("--collision", type=Path, default=Path("artifacts/orca_formal/CAP12_convex_decomposition"))
    parser.add_argument("--reference", type=Path, default=Path("artifacts/orca_formal/CAP10_sequence_mount_ik/kinematic_reference.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--axis-aligned", action="store_true")
    parser.add_argument("--installation-lift", type=float, default=0.)
    parser.add_argument("--bottle-collision", type=Path)
    parser.add_argument("--thread-contacts", choices=("mesh", "analytic"), default="mesh")
    args = parser.parse_args()
    if not np.isfinite(args.installation_lift) or not 0 <= args.installation_lift <= .006:
        parser.error("installation-lift must be in [0,.006] metres")
    _, report = build(args.scene, args.collision, args.reference, args.output,
                      args.axis_aligned, args.installation_lift, args.bottle_collision, args.thread_contacts)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
