"""Check unscaled HIC grasp reachability and native dynamics before PPO.

One rigid object-frame alignment is applied to both object and fingertips.
Object translation is calibrated inside the fixed-wrist workspace; no individual
finger endpoint offsets or object resizing are allowed.
"""

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from diagnose_orca_grasp import contact_state, model_indices
from prepare_hic_orca_object import digest, compiled_vertices


def cube_penetration(model, data):
    gid = model.geom("cube").id
    depth = 0.
    for c in data.contact[:data.ncon]:
        if gid in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == gid else c.geom1
            if model.geom_contype[other] == 1:
                depth = max(depth, -float(c.dist))
    return depth


def native_rollout(model, cube, quat, reference, seed, seconds=40., initial_ctrl=None):
    qids, cube_id, _ = model_indices(model)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    initial = np.array([-.25] + [0.] * 16) if initial_ctrl is None else np.asarray(initial_ctrl)
    spawn = cube + rng.uniform(-1, 1, 3) * [.002, .002, 0]
    data.qpos[qids] = np.clip(initial + rng.normal(0, .002, 17), *model.actuator_ctrlrange.T)
    data.qpos[cube_id:cube_id + 7] = [*spawn, *quat]
    data.ctrl[:] = initial
    rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    if int(model.geom("cube").type[0]) == mujoco.mjtGeom.mjGEOM_BOX:
        from itertools import product
        vertices = np.array(list(product(*zip(-model.geom("cube").size, model.geom("cube").size))))
    else:
        vertices = compiled_vertices(model)
    lowest = np.min((vertices @ rotation.T)[:, 2])
    support = spawn + [0, 0, lowest - .004]
    data.mocap_pos[0] = support
    mujoco.mj_forward(model, data)
    reset_penetration = cube_penetration(model, data)
    records = []
    hold = best = 0
    dt = .05
    for step in range(round(seconds / dt)):
        t = (step + 1) * dt
        base = reference[min(step, len(reference) - 1)]
        data.ctrl[:] = np.clip(base, data.ctrl - .1, data.ctrl + .1)
        data.mocap_pos[0] = support - [0, 0, np.clip((t - 2) * .2, 0, .2)]
        for _ in range(round(dt / model.opt.timestep)):
            mujoco.mj_step(model, data)
        fingers, forbidden, force = contact_state(model, data)
        finite = np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        alive = finite and data.qpos[cube_id + 2] >= spawn[2] - .055
        valid = alive and t >= 3 and data.qpos[cube_id + 2] > spawn[2] - .01 and fingers[0] and fingers[1:].any() and not forbidden
        hold = hold + 1 if valid else 0
        best = max(best, hold)
        records.append(dict(qpos=data.qpos.copy(), qvel=data.qvel.copy(), ctrl=data.ctrl.copy(),
                            mocap=data.mocap_pos.copy(), alive=alive, hold=hold * dt,
                            contacts=int(fingers.sum()), thumb=int(fingers[0]), forbidden=int(forbidden),
                            penetration=cube_penetration(model, data), force=force))
        if not alive:
            break
    trace = {k: np.asarray([r[k] for r in records]) for k in records[0]}
    report = dict(seed=seed, reset_penetration_m=reset_penetration,
                  max_penetration_m=float(trace["penetration"].max()),
                  max_abs_qvel=float(np.abs(trace["qvel"]).max()),
                  best_hold_sec=best * dt, duration_sec=len(records) * dt,
                  success30=bool(best * dt >= 30 and valid and len(records) == round(seconds / dt)),
                  max_force_n=float(trace["force"].max()))
    return trace, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--model", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--starts", type=int, default=12)
    p.add_argument("--fit-rotation", action="store_true")
    p.add_argument("--yaw-only", action="store_true")
    p.add_argument("--collision-weight", type=float, default=0.)
    p.add_argument("--middle-weight", type=float, default=1.)
    p.add_argument("--contact-fingers-only", action="store_true")
    p.add_argument("--phase-start", type=int)
    p.add_argument("--joint-margin", type=float, default=0.)
    p.add_argument("--object-up-axis", choices=("positive-y", "negative-y"), default="positive-y")
    p.add_argument("--target-source", choices=("dexmv-fitted", "human-tracking"), default="dexmv-fitted")
    args = p.parse_args()
    if args.starts < 1 or not np.isfinite([args.collision_weight, args.middle_weight]).all() or min(args.collision_weight, args.middle_weight) < 0:
        p.error("Invalid start count or residual weights")
    if args.yaw_only and not args.fit_rotation:
        p.error("yaw-only requires fit-rotation")
    if args.phase_start is not None and not 0 <= args.phase_start < 95:
        p.error("phase-start must be in [0,95)")
    if args.contact_fingers_only and args.middle_weight != 0:
        p.error("Frozen noncontact fingers require middle-weight=0")
    if not np.isfinite(args.joint_margin) or not 0 <= args.joint_margin <= .15:
        p.error("joint-margin must lie in [0,.15] radians")
    source_model = args.model or args.source / "cube.xml"
    packet = args.source / "source_correspondence.npz"
    manifest = json.loads((args.source / "manifest.json").read_text())
    for name, expected in manifest["generated_sha256"].items():
        assert digest(args.source / name) == expected
    model = mujoco.MjModel.from_xml_path(str(source_model))
    data = mujoco.MjData(model)
    qids, cube_id, tip_ids = model_indices(model)
    tips = tip_ids[:3]
    selected = np.arange(10 if args.contact_fingers_only else 7, 17)
    n = len(selected)
    initial = np.array([-.25] + [0.] * 16)
    with np.load(packet) as z:
        local = (z["dexmv_fitted_object_local"][:96, [3, 6, 9]].copy()
                 if args.target_source == "dexmv-fitted"
                 else z["human_joints_object_local"][:96, [4, 8, 12]].copy())
        source_time = z["time"][:96].copy()
    # HIC opposing faces are +/-object z. Orient them across the Orca grasp
    # (world +/-y) and map the proximal direction toward its palm (world -z).
    rotation = (np.array([[-1., 0, 0], [0, 0, 1.], [0, 1., 0]]) if args.object_up_axis == "positive-y"
                else np.array([[1., 0, 0], [0, 0, 1.], [0, -1., 0]]))
    initial_rotation = rotation.copy()
    assert np.isclose(np.linalg.det(rotation), 1)
    quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    low = np.r_[model.actuator_ctrlrange[selected, 0], [.015, -.015, .235]]
    high = np.r_[model.actuator_ctrlrange[selected, 1], [.100, .065, .330]]
    low[:n] += args.joint_margin
    high[:n] -= args.joint_margin
    if args.fit_rotation:
        limits = [np.pi] if args.yaw_only else [np.pi / 2] * 3
        low, high = np.r_[low, -np.array(limits)], np.r_[high, limits]
    origin = np.array([.055, .025, .280])
    point_weights = np.array([1., 1., args.middle_weight])[:, None]
    cube_gid = model.geom("cube").id
    def penalties():
        result = np.zeros(model.ngeom)
        mujoco.mj_forward(model, data)
        for c in data.contact[:data.ncon]:
            if cube_gid in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == cube_gid else c.geom1
                if model.geom_contype[other] == 1:
                    result[other] = max(result[other], -float(c.dist) - .0005)
        return args.collision_weight * result
    def decode(x):
        if args.fit_rotation:
            change = Rotation.from_euler("z", x[n + 3]) if args.yaw_only else Rotation.from_rotvec(x[n + 3:])
            rot = change.as_matrix() @ initial_rotation
        else:
            rot = initial_rotation
        quaternion = Rotation.from_matrix(rot).as_quat()[[3, 0, 1, 2]]
        return x[n:n + 3], rot, quaternion
    def endpoint_residual(x):
        ctrl = initial.copy()
        ctrl[selected] = x[:n]
        data.qpos[qids] = ctrl
        mujoco.mj_kinematics(model, data)
        position, rot, quaternion = decode(x)
        data.qpos[cube_id:cube_id + 7] = [*position, *quaternion]
        target = local[-1] @ rot.T + position
        result = np.r_[((data.geom_xpos[tips] - target) * point_weights).ravel(),
                       .001 * x[:n], .01 * (position - origin), .001 * x[n + 3:]]
        return np.r_[result, penalties()] if args.collision_weight else result
    rng = np.random.default_rng(101)
    candidates = []
    for i in range(args.starts):
        x0 = np.r_[rng.uniform(low[:n], high[:n]), origin]
        if args.fit_rotation:
            x0 = np.r_[x0, rng.uniform(-np.pi, np.pi, 1) if args.yaw_only else rng.uniform(-.3, .3, 3)]
        fit = least_squares(endpoint_residual, x0, bounds=(low, high), max_nfev=250,
                            ftol=1e-9, xtol=1e-9, gtol=1e-9)
        error = endpoint_residual(fit.x)[:9]
        position, rot, quaternion = decode(fit.x)
        data.qpos[cube_id:cube_id + 7] = [*position, *quaternion]
        mujoco.mj_forward(model, data)
        candidates.append(dict(start=i, x=fit.x.tolist(), marker_error_m=float(np.linalg.norm(error)),
                               penetration_m=cube_penetration(model, data)))
        print(json.dumps(candidates[-1]), flush=True)
    # Select by pre-registered kinematic feasibility, not reward or test success.
    best = min(candidates, key=lambda c: c["marker_error_m"] + 5 * max(c["penetration_m"] - .001, 0))
    cube, rotation, quat = decode(np.array(best["x"]))
    data.qpos[cube_id:cube_id + 7] = [*cube, *quat]
    targets = local @ rotation.T + cube
    previous = np.array(best["x"])[:n]
    controls = np.tile(initial, (len(local), 1))
    errors = np.zeros((len(local), 3))
    for frame in reversed(range(len(local))):
        def residual(x):
            q = initial.copy()
            q[selected] = x
            data.qpos[qids] = q
            mujoco.mj_kinematics(model, data)
            terms = np.r_[((data.geom_xpos[tips] - targets[frame]) * point_weights).ravel(), .002 * (x - previous)]
            return np.r_[terms, penalties()] if args.collision_weight else terms
        fit = least_squares(residual, previous, bounds=(low[:n], high[:n]), max_nfev=150,
                            ftol=1e-8, xtol=1e-8, gtol=1e-8)
        residual(fit.x)
        errors[frame] = np.linalg.norm(data.geom_xpos[tips] - targets[frame], axis=1)
        controls[frame, selected] = fit.x
        previous = fit.x
    times = np.arange(31) * .05
    start_frame = args.phase_start or 0
    phase_time = source_time[start_frame:] - source_time[start_frame]
    ctrl = np.stack([np.interp(times / 1.5, phase_time / phase_time[-1], controls[start_frame:, k])
                     for k in range(17)], axis=1)
    # The inherited open reset requires a documented acquisition transition.
    if args.phase_start is None:
        ctrl = initial + (times / times[-1])[:, None] * (ctrl - initial)
    else:
        initial = ctrl[0].copy()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "probe_source.py").write_bytes(Path(__file__).read_bytes())
    tree = ET.parse(source_model)
    for asset in tree.findall("asset/mesh"):
        file = Path(asset.get("file"))
        if not file.is_absolute():
            asset.set("file", str((source_model.parent / file).resolve()))
    tree.find(".//body[@name='cube']").set("pos", " ".join(map(str, cube)))
    tree.find(".//body[@name='cube']").set("quat", " ".join(map(str, quat)))
    lowest = np.min((compiled_vertices(model) @ rotation.T)[:, 2])
    tree.find(".//body[@name='cube_support']").set("pos", " ".join(map(str, cube + [0, 0, lowest - .004])))
    tree.write(args.output / "cube.xml", encoding="unicode")
    proxy_path = source_model.parent / "object_proxy_report.json"
    if proxy_path.exists():
        proxy = json.loads(proxy_path.read_text())
        proxy.update(parent_model_sha256=digest(source_model), model_sha256=digest(args.output / "cube.xml"))
        (args.output / "object_proxy_report.json").write_text(json.dumps(proxy, indent=2) + "\n")
    np.savez_compressed(args.output / "reference.npz", ctrl=ctrl, time=times,
                        raw_ik_ctrl=controls, target_tips=targets, source_local_tips=local,
                        object_position=cube, object_quaternion=quat, object_to_world_rotation=rotation,
                        source_time=source_time, spatial_scale=np.array(1.), object_extents_m=[.053] * 3,
                        target_source=np.array(args.target_source),
                        initial_ctrl=initial, source_frame_start=np.array(start_frame))
    from orca_residual_controls import load_reference
    physical_reference, _ = load_reference(args.output / "reference.npz", *model.actuator_ctrlrange.T,
                                           initial, .05, 800, .1)
    checks = []
    for seed in range(3):
        trace, check = native_rollout(model, cube, quat, physical_reference, seed, initial_ctrl=initial)
        checks.append(check)
        np.savez_compressed(args.output / f"native_seed{seed}.npz", **trace)
        print("NATIVE", json.dumps(check), flush=True)
    report = dict(method="one rigid object-frame transform plus bounded IK; no endpoint-specific offsets",
                  trainable=False, source_sha256=digest(packet), source_model_sha256=digest(source_model),
                  model_sha256=digest(args.output / "cube.xml"), reference_sha256=digest(args.output / "reference.npz"),
                  spatial_scale=1., source_frames=[start_frame, 95], source_duration_sec=float(phase_time[-1]),
                  target_source=args.target_source,
                  target_duration_sec=1.5, acquisition_blend_sec=1.5 if args.phase_start is None else 0.,
                  contact_fingers_only=args.contact_fingers_only,
                  joint_margin_rad=args.joint_margin,
                  object_up_axis=args.object_up_axis,
                  selected_actuators=[model.actuator(i).name for i in selected],
                  fit_rotation=args.fit_rotation, collision_weight=args.collision_weight, middle_weight=args.middle_weight,
                  yaw_only=args.yaw_only, command=sys.argv, source_script_sha256=digest(Path(__file__)),
                  cube=cube.tolist(), cube_quaternion=quat.tolist(), object_to_world_rotation=rotation.tolist(),
                  candidates=candidates, selected_start=best["start"],
                  mean_marker_error_mm=(errors.mean(axis=0) * 1000).tolist(),
                  endpoint_marker_error_mm=(errors[-1] * 1000).tolist(),
                  native_checks=checks,
                  note="Kinematic/dynamic feasibility probe only; not PPO, not a full video transport reproduction")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("SUMMARY", json.dumps({k: v for k, v in report.items() if k != "candidates"}), flush=True)


if __name__ == "__main__":
    main()
