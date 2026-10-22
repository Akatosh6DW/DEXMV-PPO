#!/usr/bin/env python3
"""Fixed bottle and passive threaded cap; native reachability diagnostics."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/orca30/confirmatory_f01/bc_seed2/model.xml"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PITCH = .002 / (2 * np.pi)


def vec(values):
    return " ".join(f"{x:.10g}" for x in values)


def build_scene(output, axis=(1., 0., 0.), center=(.055, .025, .265)):
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    root.set("model", "Orca fixed bottle passive screw cap v1")
    root.find("option").set("iterations", "40")
    root.find("option").set("timestep", ".002")
    world = root.find("worldbody")
    for name in ("cube", "cube_support"):
        world.remove(world.find(f"body[@name='{name}']"))
    # Convex polygon avoids the unsupported MJX box-cylinder collision pair.
    vertices = [[.019 * np.cos(a), .019 * np.sin(a), z]
                for z in (-.016, .016) for a in np.arange(16) * 2 * np.pi / 16]
    ET.SubElement(root.find("asset"), "mesh", name="cap_convex", vertex=vec(np.ravel(vertices)))
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    rot, _ = Rotation.align_vectors(axis[None, :], np.array([[0., 0., 1.]]))
    q = rot.as_quat()
    bottle = ET.SubElement(world, "body", name="fixed_bottle", pos=vec(center),
                           quat=vec(q[[3, 0, 1, 2]]))
    ET.SubElement(bottle, "geom", name="bottle_body", type="box", size=".024 .024 .040",
                  pos="0 0 -.072", contype="4", conaffinity="1", group="1",
                  rgba=".12 .65 .52 1")
    ET.SubElement(bottle, "geom", name="bottle_neck", type="box", size=".012 .012 .007",
                  pos="0 0 -.024", contype="4", conaffinity="1", group="1",
                  rgba=".12 .65 .52 1")
    ET.SubElement(bottle, "geom", name="bottle_clamp", type="box", size=".030 .030 .008",
                  pos="0 0 -.100", contype="4", conaffinity="1", group="1",
                  rgba=".25 .27 .30 1")
    cap = ET.SubElement(bottle, "body", name="cap")
    ET.SubElement(cap, "joint", name="cap_lift", type="slide", axis="0 0 1",
                  limited="false", damping=".02", frictionloss="0", armature=".00001")
    ET.SubElement(cap, "joint", name="cap_turn", type="hinge", axis="0 0 1",
                  limited="true", range="-.15 12.56637061", damping=".0001",
                  frictionloss=".001", armature=".00001")
    ET.SubElement(cap, "geom", name="cap", type="mesh", mesh="cap_convex", mass=".015",
                  contype="2", conaffinity="1", group="1", friction="1.35 .025 .004",
                  rgba=".95 .40 .10 1")
    ET.SubElement(cap, "geom", name="cap_angle_marker", type="box", size=".014 .002 .0003",
                  pos="0 0 .0164", contype="0", conaffinity="0", mass="0", group="1",
                  rgba="1 1 1 1")
    eq = ET.SubElement(root, "equality")
    ET.SubElement(eq, "joint", name="thread_pitch", joint1="cap_turn", joint2="cap_lift",
                  polycoef=vec([0, 1 / PITCH, 0, 0, 0]), solref=".004 1", solimp=".9999 .9999 .001")
    ET.indent(tree)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="unicode")
    return mujoco.MjModel.from_xml_path(str(output))


def indices(model):
    return (model.jnt_qposadr[model.actuator_trnid[:, 0]],
            model.joint("cap_turn").qposadr[0], model.joint("cap_lift").qposadr[0],
            [model.geom(f"orca_{f}_tip_ball_proxy").id for f in FINGERS])


def fit_grip(model, angle=0., start=None, preload=.001):
    data = mujoco.MjData(model)
    qids, turn, lift, tips = indices(model)
    selected = np.array([0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    ctrl = np.zeros(17) if start is None else np.array(start).copy()
    if start is None:
        ctrl[[0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] = [-.25, 0, .5, 1.5, 0, .5, 1.5, .2, .5, .4, .4]
    mujoco.mj_forward(model, data)
    bid = model.body("fixed_bottle").id
    center, rotation = data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()
    # World-Y opposition reproduces the video grasp's thumb/finger arrangement.
    axis = rotation[:, 2]
    radial = np.array([0., 1., 0.])
    radial -= axis * np.dot(axis, radial)
    radial /= np.linalg.norm(radial)
    twist = Rotation.from_rotvec(axis * angle).as_matrix()
    target = []
    for i in range(3):
        v = radial * (1 if i == 0 else -1) * (.019 + model.geom_size[tips[i], 0] - preload)
        v += axis * (0 if i == 0 else (-.008 if i == 1 else .008))
        target.append(center + twist @ v + axis * PITCH * angle)
    target = np.asarray(target)
    lo, hi = model.actuator_ctrlrange[selected].T
    anchor = ctrl[selected].copy()

    def residual(x):
        ctrl[selected] = x
        data.qpos[qids] = ctrl
        data.qpos[turn], data.qpos[lift] = angle, PITCH * angle
        mujoco.mj_forward(model, data)
        return np.r_[(data.geom_xpos[tips[:3]] - target).ravel(), .0002 * (x - anchor)]

    result = least_squares(residual, np.clip(ctrl[selected], lo + 1e-7, hi - 1e-7),
                           bounds=(lo, hi), max_nfev=100)
    error = np.linalg.norm(residual(result.x)[:9])
    ctrl[selected] = result.x
    return ctrl.copy(), float(error)


def contacts(model, data):
    cap = model.geom("cap").id
    finger = np.zeros(5, bool)
    forbidden = False
    force_max = 0.
    for c in range(data.ncon):
        hit = data.contact[c]
        if cap not in (hit.geom1, hit.geom2) or hit.dist > 0:
            continue
        other = hit.geom2 if hit.geom1 == cap else hit.geom1
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, c, force)
        if force[0] <= 1e-5:
            continue
        name = model.geom(other).name
        matched = np.array([name.startswith(f"orca_{f}_") for f in FINGERS])
        finger |= matched
        forbidden |= not matched.any()
        force_max = max(force_max, float(force[0]))
    return finger, forbidden, force_max


def native_rollout(model, controls, seconds=12.):
    data = mujoco.MjData(model)
    qids, turn, lift, _ = indices(model)
    data.qpos[qids[0]] = -.25
    data.ctrl[0] = -.25
    rows = []
    for step in range(round(seconds / .05)):
        target = controls[min(step, len(controls) - 1)]
        data.ctrl[:] = np.clip(target, data.ctrl - .1, data.ctrl + .1)
        mujoco.mj_step(model, data, nstep=round(.05 / model.opt.timestep))
        finger, forbidden, force = contacts(model, data)
        rows.append(dict(qpos=data.qpos.copy(), qvel=data.qvel.copy(), ctrl=data.ctrl.copy(),
                         angle=data.qpos[turn], lift=data.qpos[lift],
                         thread_error=data.qpos[lift] - PITCH * data.qpos[turn],
                         fingers=finger, forbidden=forbidden, force=force))
    trace = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    valid = trace["fingers"][:, 0] & trace["fingers"][:, 1:].any(axis=1) & ~trace["forbidden"]
    summary = dict(final_degrees=float(np.rad2deg(trace["angle"][-1])),
                   max_degrees=float(np.rad2deg(trace["angle"].max())),
                   paired_contact_fraction=float(valid.mean()),
                   forbidden_fraction=float(trace["forbidden"].mean()),
                   max_thread_error_m=float(abs(trace["thread_error"]).max()),
                   max_force_n=float(trace["force"].max()))
    return trace, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--axis", type=float, nargs=3, default=[1, 0, 0])
    p.add_argument("--center", type=float, nargs=3, default=[.055, .025, .265])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = build_scene(args.output / "model.xml", args.axis, args.center)
    controls = []
    ctrl, error = fit_grip(model)
    for alpha in np.linspace(0, 1, 40):
        initial = np.zeros(17)
        initial[0] = -.25
        controls.append(initial * (1 - alpha) + ctrl * alpha)
    errors = [error]
    for angle in np.linspace(0, np.pi / 2, 120):
        ctrl, error = fit_grip(model, angle, ctrl)
        controls.append(ctrl.copy())
        errors.append(error)
    controls = np.asarray(controls)
    trace, summary = native_rollout(model, controls)
    zero = np.zeros((1, 17))
    zero[0, 0] = -.25
    _, negative = native_rollout(model, zero)
    np.savez_compressed(args.output / "native_trace.npz", **trace)
    np.savez_compressed(args.output / "ik_controls.npz", ctrl=controls, errors=errors)
    report = dict(kind="procedural_IK_not_video_not_PPO", axis=args.axis, center=args.center,
                  summary=summary, negative_control=negative, mean_ik_error_m=float(np.mean(errors)),
                  max_ik_error_m=float(np.max(errors)))
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
