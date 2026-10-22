#!/usr/bin/env python3
"""Bounded fingertip IK and native physical reachability checks, not RL."""

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/orca_hand/xmls/orcahand_description/v2/scene_right_mjx_grasp_cube.xml"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def model_indices(model):
    qids = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    cube_qid = model.joint("cube_freejoint").qposadr[0]
    tip_ids = [model.geom(f"orca_{f}_tip_ball_proxy").id for f in FINGERS]
    return qids, cube_qid, tip_ids


def fit_pose(model, cube, preload=0.001, wrist=0.0, opposition=1):
    data = mujoco.MjData(model)
    qids, cube_id, tips = model_indices(model)
    selected = np.array([10, 11, 12, 7, 8, 9, 13, 14, 15, 16])
    low, high = model.actuator_ctrlrange[selected].T
    radii = model.geom_size[tips, 0]
    target = np.repeat(np.asarray(cube)[None, :], 3, axis=0)
    target[0, 1] += opposition * (0.018 + radii[0] - preload)
    target[1:, 1] -= opposition * (0.018 + radii[1:3] - preload)
    target[1, 0] -= 0.010
    target[2, 0] += 0.010
    ctrl = np.zeros(model.nu)
    ctrl[0] = wrist

    def residual(x):
        ctrl[selected] = x
        data.qpos[qids] = ctrl
        data.qpos[cube_id:cube_id + 3] = cube
        mujoco.mj_forward(model, data)
        return np.concatenate(((data.geom_xpos[tips[:3]] - target).ravel(),
                               0.0003 * x))

    start = np.array([0, .9, .9, 0, .9, .9, .2, .7, .8, .8])
    result = least_squares(residual, np.clip(start, low + 1e-6, high - 1e-6),
                           bounds=(low, high), max_nfev=150,
                           ftol=1e-8, xtol=1e-8, gtol=1e-8)
    error = float(np.linalg.norm(residual(result.x)[:9]))
    ctrl[selected] = result.x
    return ctrl.copy(), error


def contact_state(model, data):
    cube_id = model.geom("cube").id
    fingers = np.zeros(5, dtype=bool)
    forbidden = False
    max_force = 0.0
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        if cube_id not in (contact.geom1, contact.geom2) or contact.dist > 0:
            continue
        other = contact.geom2 if contact.geom1 == cube_id else contact.geom1
        name = model.geom(other).name
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, contact_id, force)
        if force[0] <= 1e-5:
            continue
        max_force = max(max_force, float(force[0]))
        for i, finger in enumerate(FINGERS):
            fingers[i] |= name.startswith(f"orca_{finger}_")
        forbidden |= not any(name.startswith(f"orca_{f}_") for f in FINGERS)
    return fingers, forbidden, max_force


def rollout(model, cube, ctrl, seconds=36.0, close_sec=1.5, trace=False, phase=None,
            control_sequence=None):
    data = mujoco.MjData(model)
    qids, cube_id, _ = model_indices(model)
    data.qpos[qids] = 0
    data.qpos[qids[0]] = ctrl[0]
    data.qpos[cube_id:cube_id + 7] = [*cube, 1, 0, 0, 0]
    support = np.asarray(cube) - [0, 0, .022]
    data.mocap_pos[0] = support
    mujoco.mj_forward(model, data)
    consecutive = best = 0
    samples = []
    peak_force = 0.0
    for step in range(round(seconds / model.opt.timestep)):
        t = step * model.opt.timestep
        alpha = np.clip(t / close_sec, 0, 1)
        if phase is not None:
            alpha = np.interp(alpha, np.linspace(0, 1, len(phase)), phase)
        data.ctrl[:] = alpha * ctrl
        if control_sequence is not None:
            data.ctrl[:] = [np.interp(t, np.arange(len(control_sequence)) * .05, control_sequence[:, k])
                            for k in range(model.nu)]
        data.ctrl[0] = ctrl[0]
        if t >= 2.0:
            data.mocap_pos[0] = support - [0, 0, min((t - 2) * .2, .2)]
        mujoco.mj_step(model, data)
        fingers, forbidden, force = contact_state(model, data)
        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        valid = (t >= 3 and finite and fingers[0] and fingers[1:].any()
                 and not forbidden and data.qpos[cube_id + 2] > cube[2] - .01)
        consecutive = consecutive + 1 if valid else 0
        best = max(best, consecutive)
        peak_force = max(peak_force, force)
        if trace and step % 5 == 0:
            samples.append(dict(time=float(data.time), qpos=data.qpos.copy().tolist(),
                                ctrl=data.ctrl.copy().tolist(),
                                mocap_pos=data.mocap_pos.copy().tolist(),
                                fingers=fingers.tolist(), forbidden=bool(forbidden),
                                valid=bool(valid)))
        if not finite or data.qpos[cube_id + 2] < cube[2] - .055:
            break
    return dict(hold_sec=best * model.opt.timestep, end_sec=float(data.time),
                final_cube=data.qpos[cube_id:cube_id + 3].tolist(),
                max_contact_force=peak_force, trace=samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/orca30/geometry_o01")
    parser.add_argument("--seconds", type=float, default=36)
    parser.add_argument("--expanded", action="store_true")
    parser.add_argument("--opposition", type=int, choices=(-1, 1), default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    started = time.monotonic()
    records = []
    import itertools
    xs = (.055, .065, .075)
    ys = (.010, .025, .040, .055) if args.expanded else (.030, .040, .050)
    zs = (.250, .265, .280, .295) if args.expanded else (.250, .260, .270)
    wrists = (-.5, -.25, .25) if args.expanded else (0.0,)
    for wrist, x, y, z in itertools.product(wrists, xs, ys, zs):
        cube = [x, y, z]
        ctrl, error = fit_pose(model, cube, wrist=wrist, opposition=args.opposition)
        result = rollout(model, cube, ctrl, args.seconds)
        result.pop("trace")
        result.update(cube=cube, ctrl=ctrl.tolist(), ik_error_m=error)
        records.append(result)
        print(json.dumps({k: result[k] for k in ("cube", "hold_sec", "end_sec", "ik_error_m")}), flush=True)
    best = max(records, key=lambda r: (r["hold_sec"], r["end_sec"], -r["ik_error_m"]))
    detailed = rollout(model, best["cube"], np.array(best["ctrl"]), args.seconds, trace=True)
    report = dict(kind="development_native_geometry_not_PPO", scene=str(SCENE),
                  opposition=args.opposition,
                  elapsed_sec=time.monotonic() - started, records=records, best=best)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    (args.output / "best_trace.json").write_text(json.dumps(detailed))
    print("BEST", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
