#!/usr/bin/env python3
"""Transfer object-relative DexMV fingertip paths to a calibrated Orca grasp."""

import hashlib
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from diagnose_orca_grasp import SCENE, model_indices, rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-weight", type=float, default=0)
    parser.add_argument("--acquisition-sec", type=float, default=.4)
    parser.add_argument("--output", type=Path, default=Path("artifacts/orca30/dexmv_trajectory_o16"))
    args = parser.parse_args()
    source = Path("artifacts/cube20/hic20/dexmv_retarget/adroit_cube.npz")
    geometry = Path("artifacts/orca30/geometry_o04/report.json")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source) as z:
        points = z["fitted_positions"][:96, [3, 6, 9]]
        objects = z["object_poses"][:96]
    local = np.einsum("tji,tkj->tki", objects[:, :3, :3], points - objects[:, None, :3, 3])
    best = json.loads(geometry.read_text())["best"]
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    qids, cube_qid, tip_ids = model_indices(model)
    endpoint = np.array(best["ctrl"])
    data.qpos[qids] = endpoint
    data.qpos[cube_qid:cube_qid + 3] = best["cube"]
    mujoco.mj_forward(model, data)
    target_end = data.geom_xpos[tip_ids[:3]].copy()
    collision_gids = [i for i in range(model.ngeom)
                      if model.geom(i).name.startswith(("orca_thumb_", "orca_index_", "orca_middle_", "orca_palm_"))]
    cube_gid = model.geom("cube").id
    distance_buffer = np.zeros(6)
    def distances():
        return np.array([mujoco.mj_geomDistance(model, data, i, cube_gid, .1, distance_buffer)
                         for i in collision_gids])
    allowed_penetration = np.minimum(distances(), 0)
    a = local[-1] - local[-1].mean(0)
    b = target_end - target_end.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    targets = target_end[None, ...] + ((local - local[-1]) @ rotation.T) * (.036 / .053)
    selected = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    low, high = model.actuator_ctrlrange[selected].T
    controls = np.repeat(endpoint[None], len(targets), axis=0)
    previous = endpoint[selected].copy()
    errors = []
    # Work backward from the verified endpoint so redundant IK solutions do
    # not jump between branches across adjacent video frames.
    for frame in reversed(range(len(targets))):
        target = targets[frame]
        def residual(x):
            ctrl = endpoint.copy()
            ctrl[selected] = x
            data.qpos[qids] = ctrl
            mujoco.mj_forward(model, data)
            terms = [(data.geom_xpos[tip_ids[:3]] - target).ravel(), .002 * (x - previous)]
            if args.collision_weight:
                terms.append(args.collision_weight * np.minimum(distances() - allowed_penetration, 0))
            return np.concatenate(terms)
        fit = least_squares(residual, np.clip(previous, low + 1e-8, high - 1e-8),
                            bounds=(low, high), max_nfev=100,
                            ftol=1e-8, gtol=1e-8, xtol=1e-8)
        controls[frame, selected] = fit.x
        errors.append(float(np.linalg.norm(residual(fit.x)[:9])))
        previous = fit.x
    initial = np.zeros(17)
    initial[0] = endpoint[0]
    times = np.arange(31) * .05
    reference = np.stack([np.interp(times / 1.5, np.linspace(0, 1, len(controls)), controls[:, k])
                          for k in range(17)], axis=1)
    if args.acquisition_sec <= 0:
        raise ValueError("acquisition duration must be positive")
    blend = np.clip(times / args.acquisition_sec, 0, 1)[:, None]
    reference = initial + blend * (reference - initial)
    np.savez_compressed(output / "reference.npz", ctrl=reference, time=times,
                        raw_ik_ctrl=controls, source_local_tips=local, target_tips=targets)
    result = rollout(model, best["cube"], endpoint, seconds=40, trace=True,
                     control_sequence=reference)
    (output / "native_trace.json").write_text(json.dumps(result))
    result.pop("trace")
    report = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest(),
                  method="object-relative 3D thumb/index/middle paths + calibrated endpoint + bounded IK",
                  source_frames=[0, 95], source_duration_sec=95/30,
                  target_duration_sec=1.5, acquisition_blend_sec=args.acquisition_sec,
                  mean_ik_error_m=float(np.mean(errors)), max_ik_error_m=max(errors),
                  collision_weight=args.collision_weight,
                  control_range=np.ptp(reference, axis=0).tolist(),
                  native_result=result)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
