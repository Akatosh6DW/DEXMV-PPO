"""Fit source cylinder thumb/index/middle positions with palmar-facing contact."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from diagnose_orca_grasp import model_indices
from orca_cylinder_s02_env import cylinder_xml
from prepare_cylinder_s02_source import digest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/orca_cylinder_s02/source/selection96/selected_segment.npz"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--starts", type=int, default=16)
    p.add_argument("--normal-weight", type=float, default=.02)
    p.add_argument("--clearance", type=float, default=0.)
    p.add_argument("--aligned-wrist", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    xml = cylinder_xml()
    (args.output / "model.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    qids, oid, _ = model_indices(model)
    gid = model.geom("cube").id
    pads = np.array([model.geom(f"orca_{f}_tip_proxy_fit_{n}").id
                     for f, n in (("thumb", "02"), ("index", "03"), ("middle", "03"))])
    bodies = model.geom_bodyid[pads]
    radii = model.geom_size[pads, 0]
    chosen = np.r_[0, np.arange(7, 17)]
    n = len(chosen)
    initial = np.r_[-.25, np.zeros(16)]
    with np.load(SOURCE) as src:
        local = src["fingertips_centered"][:, :3] * (.053 / src["source_extents_m"])
        source_time = src["time"]
    radial = np.linalg.norm(local[:, :, :2], axis=-1)
    outward = np.zeros_like(local)
    outward[:, :, :2] = local[:, :, :2] / radial[:, :, None]
    near = np.clip((.02 - np.abs(radial - .0265)) / .012, 0, 1)
    targets = local.copy()
    targets[:, :, :2] += near[:, :, None] * (outward[:, :, :2] * (.0265 + radii - .0006)[None, :, None] - local[:, :, :2])
    # Source Z is the cylinder axis; target geom's axis is body Y.
    source_to_body = Rotation.from_euler("x", np.pi / 2).as_matrix()
    targets = targets @ source_to_body.T
    inward = -outward @ source_to_body.T
    base_rotation = np.array([[1., 0, 0], [0, 0, 1.], [0, -1., 0]])
    low = np.r_[model.actuator_ctrlrange[chosen, 0] + .04, [-.04, -.10, .19], -np.pi]
    high = np.r_[model.actuator_ctrlrange[chosen, 1] - .04, [.12, .08, .34], np.pi]
    allowed = {i for i in range(model.ngeom) if any((model.geom(i).name or "").startswith(f"orca_{f}_tip_") for f in ("thumb", "index", "middle"))}
    if args.clearance:
        allowed = set(pads) | {model.geom(f"orca_{f}_tip_proxy_fit_02").id for f in ("index", "middle")}
        physical = model.geom_contype == 1
        model.geom_margin[physical] = args.clearance
        model.geom_gap[physical] = args.clearance

    def place(q, pos, rot):
        data.qpos[qids] = initial
        data.qpos[qids[chosen]] = q
        data.qpos[oid:oid+7] = np.r_[pos, Rotation.from_matrix(rot).as_quat()[[3, 0, 1, 2]]]
        data.mocap_pos[0] = [0, 0, -1]
        mujoco.mj_forward(model, data)

    def residual(q, pos, rot, frame):
        place(q, pos, rot)
        normal = -data.xmat[bodies].reshape(3, 3, 3)[:, :, 0]
        penalty = np.zeros(model.ngeom)
        for c in data.contact[:data.ncon]:
            if gid in c.geom:
                other = c.geom2 if c.geom1 == gid else c.geom1
                if model.geom_contype[other] == 1:
                    penalty[other] = max(penalty[other], -c.dist + (-.0008 if other in allowed else args.clearance))
        return np.r_[(data.geom_xpos[pads] - (targets[frame] @ rot.T + pos)).ravel(),
                     (args.normal_weight * near[frame, :, None] * (normal - inward[frame] @ rot.T)).ravel(),
                     8 * penalty]

    def rotation(x):
        return Rotation.from_euler("z", x[-1]).as_matrix() @ base_rotation

    def endpoint(x):
        return residual(x[:n], x[n:n+3], rotation(x), -1)

    candidates = []
    rng = np.random.default_rng(6101)
    for i in range(args.starts):
        fit = least_squares(endpoint, rng.uniform(low, high), bounds=(low, high), max_nfev=350)
        error = endpoint(fit.x)
        row = dict(start=i, x=fit.x.tolist(), cost=float(np.linalg.norm(error)),
                   tip_errors_mm=(np.linalg.norm(error[:9].reshape(3, 3), axis=1)*1000).tolist())
        candidates.append(row)
        print(json.dumps(row), flush=True)
    best = min(candidates, key=lambda x: x["cost"])
    x = np.asarray(best["x"])
    pos, rot, previous = x[n:n+3], rotation(x), x[:n]
    controls = np.tile(initial, (len(local), 1))
    errors = []
    for frame in reversed(range(len(local))):
        fit = least_squares(lambda q: np.r_[residual(q, pos, rot, frame), .0005*(q-previous)],
                            previous, bounds=(low[:n], high[:n]), max_nfev=100)
        previous = fit.x
        controls[frame, chosen] = previous
        error = residual(previous, pos, rot, frame)
        errors.append(np.linalg.norm(error[:9].reshape(3, 3), axis=1))
    times = np.arange(31) * .05
    ctrl = np.stack([np.interp(times/1.5, source_time/source_time[-1], controls[:, k]) for k in range(17)], axis=1)
    if args.aligned_wrist:
        initial[0] = ctrl[-1, 0]
    # Begin from the inherited open posture; retain source motion during closure.
    ctrl = initial + (times/1.5)[:, None] * (ctrl-initial)
    np.savez_compressed(args.output / "reference.npz", ctrl=ctrl, time=times, initial_ctrl=initial,
                        object_position=pos, object_quaternion=Rotation.from_matrix(rot).as_quat()[[3, 0, 1, 2]],
                        object_to_world_rotation=rot, object_extents_m=np.full(3, .053), spatial_scale=1.,
                        raw_ik_ctrl=controls, source_local_tips=local, target_body_tips=targets,
                        source_to_body_rotation=source_to_body, selected_actuators=chosen,
                        source_indices=np.arange(182, 272), source_sequence=96)
    (args.output / "report.json").write_text(json.dumps(dict(source=str(SOURCE), source_sha256=digest(SOURCE),
        best=best, candidates=candidates, errors_mm=(np.asarray(errors)[::-1]*1000).tolist(),
        fingers=["thumb", "index", "middle"], normal_weight=args.normal_weight,
        clearance=args.clearance, aligned_wrist=args.aligned_wrist,
        excluded="Ring stays distant; little-finger source reconstruction has up to 41.6 mm interpenetration. Not copied as a physical expert.",
        changes="Three-finger geometry, facing constraint, free wrist and shared workspace alignment; physics unchanged"), indent=2)+"\n")


if __name__ == "__main__":
    main()
