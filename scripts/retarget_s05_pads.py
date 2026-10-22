"""S05: fit distal pad positions AND facing directions to HIC opposing faces."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from diagnose_orca_grasp import model_indices
from probe_hic53_orca import cube_penetration, native_rollout
from prepare_hic_orca_object import digest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/orca_collision_refit/O20_hic53_elliptic/cube.xml"
SOURCE = ROOT / "artifacts/orca_collision_refit/O01_hic53_source_object/source_correspondence.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=16)
    parser.add_argument("--normal-weight", type=float, default=.025)
    parser.add_argument("--seconds", type=float, default=8.)
    parser.add_argument("--full-rotation", action="store_true")
    parser.add_argument("--up-axis", choices=("x", "negative-x", "y", "negative-y", "z", "negative-z"), default="y")
    parser.add_argument("--free-wrist", action="store_true")
    parser.add_argument("--duration", type=float, default=3.)
    parser.add_argument("--y-min", type=float, default=-.04)
    parser.add_argument("--nonpad-clearance", type=float, default=0.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "model.xml").write_bytes(MODEL.read_bytes())
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    qids, cube_qid, _ = model_indices(model)
    cube = model.geom("cube").id
    pads = np.array([model.geom("orca_thumb_tip_proxy_fit_02").id,
                     model.geom("orca_index_tip_proxy_fit_03").id])
    bodies = model.geom_bodyid[pads]
    allowed = set(pads) | {model.geom("orca_index_tip_proxy_fit_02").id}
    original_margin, original_gap = model.geom_margin.copy(), model.geom_gap.copy()
    if args.nonpad_clearance:
        physical_hand = model.geom_contype == 1
        model.geom_margin[physical_hand] = args.nonpad_clearance
        model.geom_gap[physical_hand] = args.nonpad_clearance
    radii = model.geom_size[pads, 0]
    chosen = np.r_[0, np.arange(10, 17)] if args.free_wrist else np.arange(10, 17)
    nq = len(chosen)
    initial = np.r_[-.25, np.zeros(16)]
    with np.load(SOURCE, allow_pickle=False) as archive:
        local = archive["human_joints_object_local"][:96, [4, 8]].copy()
        source_time = archive["time"][:96].copy()
    # Human joint centers are not skin surfaces. Preserve the in-face locations,
    # but map each center's surface distance to the robot's physical pad radius.
    signs = np.sign(local[-1, :, 2])
    targets = local.copy()
    near = np.clip((.020 - np.abs(np.abs(local[:, :, 2]) - .0265)) / .010, 0, 1)
    targets[:, :, 2] += near * (signs * (.0265 + radii - .0005) - targets[:, :, 2])
    rotations = 3 if args.full_rotation else 1
    low = np.r_[model.actuator_ctrlrange[chosen, 0] + .04, [-.03, args.y_min, .19], np.full(rotations, -np.pi)]
    high = np.r_[model.actuator_ctrlrange[chosen, 1] - .04, [.12, .09, .33], np.full(rotations, np.pi)]
    original_rotation = dict(
        y=[[-1., 0, 0], [0, 0, 1.], [0, 1., 0]],
        x=[[0., 1, 0], [0, 0, 1.], [1, 0, 0]],
        z=[[1., 0, 0], [0, 1, 0.], [0, 0, 1.]],
    )[args.up_axis.removeprefix("negative-")]
    original_rotation = np.array(original_rotation)
    if args.up_axis.startswith("negative-"):
        original_rotation = Rotation.from_euler("x", np.pi).as_matrix() @ original_rotation

    def place(q, position, rotation):
        data.qpos[qids] = initial
        data.qpos[qids[chosen]] = q
        quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        data.qpos[cube_qid:cube_qid + 7] = np.r_[position, quat]
        data.mocap_pos[0] = [0, 0, -1]
        mujoco.mj_forward(model, data)

    def penalty():
        depths = np.zeros(model.ngeom)
        for c in data.contact[:data.ncon]:
            if cube in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == cube else c.geom1
                if model.geom_contype[other] == 1:
                    clearance = args.nonpad_clearance if other not in allowed else -.0008
                    depths[other] = max(depths[other], -c.dist + clearance)
        return 8 * depths

    def residual(q, pos, rot, frame, orientation=1.):
        place(q, pos, rot)
        desired = targets[frame] @ rot.T + pos
        normals = -data.xmat[bodies].reshape(2, 3, 3)[:, :, 0]
        inward = -signs[:, None] * rot[:, 2]
        return np.r_[(data.geom_xpos[pads] - desired).ravel(),
                     args.normal_weight * orientation * (normals - inward).ravel(), penalty()]

    def decode_rotation(x):
        delta = Rotation.from_rotvec(x[nq + 3:]) if args.full_rotation else Rotation.from_euler("z", x[-1])
        return delta.as_matrix() @ original_rotation

    def endpoint(x):
        rot = decode_rotation(x)
        return residual(x[:nq], x[nq:nq + 3], rot, 95)

    rng = np.random.default_rng(5001)
    candidates = []
    for i in range(args.starts):
        start = rng.uniform(low, high)
        fit = least_squares(endpoint, start, bounds=(low, high), max_nfev=400,
                            ftol=1e-9, xtol=1e-9, gtol=1e-9)
        errors = endpoint(fit.x)
        row = dict(start=i, x=fit.x.tolist(), cost=float(np.linalg.norm(errors)),
                   position_error_mm=float(np.linalg.norm(errors[:6]) * 1000),
                   normal_error=float(np.linalg.norm(errors[6:12]) / args.normal_weight),
                   penetration_m=cube_penetration(model, data))
        candidates.append(row)
        print(json.dumps(row), flush=True)
    best = min(candidates, key=lambda c: c["cost"])
    x = np.asarray(best["x"])
    rotation = decode_rotation(x)
    position = x[nq:nq + 3]
    quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    previous = x[:nq]
    controls = np.tile(initial, (96, 1))
    for frame in reversed(range(96)):
        def objective(q):
            return np.r_[residual(q, position, rotation, frame, float(near[frame].min())),
                         .0005 * (q - previous)]
        fit = least_squares(objective, previous, bounds=(low[:nq], high[:nq]), max_nfev=100)
        controls[frame, chosen] = previous = fit.x
    times = np.arange(round(args.duration / .05) + 1) * .05
    ctrl = np.stack([np.interp(times / args.duration, source_time / source_time[-1], controls[:, k])
                     for k in range(17)], axis=1)
    initial = ctrl[0].copy()
    # Contact margins are a kinematic clearance query only, never a physics edit.
    model.geom_margin[:], model.geom_gap[:] = original_margin, original_gap
    np.savez_compressed(args.output / "reference.npz", ctrl=ctrl, time=times,
                        raw_ik_ctrl=controls, initial_ctrl=initial, object_position=position,
                        object_quaternion=quat, object_to_world_rotation=rotation,
                        object_extents_m=np.full(3, .053), spatial_scale=1.,
                        source_time=source_time, source_local_tips=local)
    reports = []
    for seed in range(5001, 5005):
        trace, report = native_rollout(model, position, quat, ctrl, seed,
                                       seconds=args.seconds, initial_ctrl=initial)
        reports.append(report)
        np.savez_compressed(args.output / f"probe_{seed}.npz", **trace)
        print("ROLLOUT", json.dumps(report), flush=True)
    (args.output / "report.json").write_text(json.dumps(dict(
        arguments=vars(args) | {"output": str(args.output)}, model=str(MODEL), source=str(SOURCE),
        source_sha256=digest(SOURCE), model_sha256=digest(MODEL),
        physical_geometry_unchanged=True, task_spawn_recalibrated=True,
        pad_geoms=[model.geom(int(g)).name for g in pads], candidates=candidates,
        best=best, probes=reports), indent=2) + "\n")


if __name__ == "__main__":
    main()
