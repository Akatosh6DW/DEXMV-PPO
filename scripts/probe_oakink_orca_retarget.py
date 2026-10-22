"""Unscaled, fixed-bottle landmark IK diagnostics; not a trained controller."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from prepare_hic_orca_object import digest

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
HUMAN_TIPS = np.array([4, 8, 12, 16, 20])
HUMAN_BASES = np.array([0, 5, 9, 17])
ROBOT_BASES = ("right_wrist", "right_i-mcp", "right_m-mcp", "right_p-mcp")


def rigid_alignment(source, target):
    source, target = np.asarray(source), np.asarray(target)
    u, _, vt = np.linalg.svd((source - source.mean(0)).T @ (target - target.mean(0)))
    correction = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ correction @ u.T
    translation = target.mean(0) - rotation @ source.mean(0)
    return rotation, translation


def fit(packet, model_path, output, stride=1):
    with np.load(packet) as source:
        frames = source["frame_id"][::stride]
        time = source["time"][::stride]
        human = source["joints_bottle"][::stride]
        cap = source["cap_in_bottle"][::stride]
    assert len(frames) > 1 and np.isfinite(human).all()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    qids = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    tips = [model.geom(f"orca_{f}_tip_ball_proxy").id for f in FINGERS]
    base_ids = [model.joint(name).id for name in ROBOT_BASES]
    low, high = model.actuator_ctrlrange.T
    if len(qids) != 17:
        raise ValueError("This diagnostic requires the unmodified 17-actuator Orca hand")
    anchor = int(np.argmin(abs(frames - 423)))
    initial = np.zeros(17)
    initial[0] = -.25

    def kinematics(ctrl):
        data.qpos[qids] = ctrl
        mujoco.mj_kinematics(model, data)
        return data.geom_xpos[tips].copy(), data.xanchor[base_ids].copy()

    _, bases = kinematics(initial)
    initial_rotation, initial_translation = rigid_alignment(human[anchor, HUMAN_BASES], bases)
    weights = np.array([1., 1., .3, .3, .3])[:, None]

    def decode_mount(x):
        rotation = Rotation.from_rotvec(x[17:20]).as_matrix() @ initial_rotation
        return rotation, initial_translation + x[20:23]

    def mount_residual(x):
        points, bases = kinematics(x[:17])
        rotation, translation = decode_mount(x)
        target = human[anchor] @ rotation.T + translation
        return np.r_[((points - target[HUMAN_TIPS]) * weights).ravel(),
                     .25 * (bases - target[HUMAN_BASES]).ravel(),
                     .001 * (x[:17] - initial), .002 * x[17:20]]

    rng = np.random.default_rng(1801)
    candidates = []
    for i in range(4):
        ctrl = initial if i == 0 else rng.uniform(low, high)
        x0 = np.r_[ctrl, np.zeros(6)]
        result = least_squares(mount_residual, x0,
                               bounds=(np.r_[low, [-.8] * 3, [-.06] * 3],
                                       np.r_[high, [.8] * 3, [.06] * 3]), max_nfev=250,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        candidates.append(dict(start=i, objective=float(np.square(mount_residual(result.x)).sum()),
                               evaluations=result.nfev, converged=bool(result.success), x=result.x.tolist()))
    selected = min(candidates, key=lambda c: c["objective"])
    rotation, translation = decode_mount(np.asarray(selected["x"]))
    targets = human[:, HUMAN_TIPS] @ rotation.T + translation
    base_targets = human[:, HUMAN_BASES] @ rotation.T + translation
    controls, errors, fitted, base_errors = [], [], [], []
    previous = np.asarray(selected["x"][:17])
    for frame in range(len(frames)):
        def residual(ctrl):
            points, bases = kinematics(ctrl)
            return np.r_[((points - targets[frame]) * weights).ravel(),
                         .05 * (bases - base_targets[frame]).ravel(), .002 * (ctrl - previous)]

        result = least_squares(residual, np.clip(previous, low + 1e-8, high - 1e-8),
                               bounds=(low, high), max_nfev=100,
                               ftol=1e-8, xtol=1e-8, gtol=1e-8)
        points, bases = kinematics(result.x)
        controls.append(result.x.copy())
        fitted.append(points)
        errors.append(np.linalg.norm(points - targets[frame], axis=1))
        base_errors.append(np.linalg.norm(bases - base_targets[frame], axis=1))
        previous = result.x.copy()
        if frame % 60 == 0:
            print(json.dumps(dict(frame=int(frames[frame]), contact_tip_errors_mm=(1000 * np.asarray(errors[-1])[:2]).tolist())), flush=True)
    controls, errors, fitted, base_errors = map(np.asarray, (controls, errors, fitted, base_errors))
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation
    near_limit = np.minimum(controls - low, high - controls) < .01
    phases = {"approach": frames < 423, "unscrew_and_remove": frames >= 423,
              "pre_transport": (frames >= 423) & (frames < 1612)}
    phase_metrics = {}
    for name, mask in phases.items():
        phase_metrics[name] = dict(frames=int(mask.sum()),
                                   mean_tip_error_mm=(1000 * errors[mask].mean(0)).tolist(),
                                   p95_tip_error_mm=(1000 * np.percentile(errors[mask], 95, axis=0)).tolist(),
                                   contact_pair_under_10mm_fraction=float((errors[mask, :2].max(1) <= .01).mean()))
    angles = np.unwrap(np.arctan2(cap[:, 1, 0], cap[:, 0, 0]))
    angles -= angles[anchor]
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "kinematic_reference.npz", ctrl=controls, time=time, source_frame_id=frames,
                        target_tips=targets, fitted_tips=fitted, errors_m=errors, base_errors_m=base_errors,
                        bottle_to_simulation=transform, source_cap_in_bottle=cap,
                        cap_to_simulation=transform @ cap, source_unwrapped_angle=angles,
                        initial_ctrl=initial, spatial_scale=1.)
    report = dict(kind="kinematic_reachability_only_not_contact_validated_not_PPO",
                  target_source="published human landmarks directly to Orca; no Adroit intermediate in this diagnostic",
                  source_packet_sha256=digest(packet), hand_model_sha256=digest(model_path),
                  source_sha256=digest(Path(__file__)), spatial_scale=1., frame_stride=stride,
                  first_contact_anchor_frame=int(frames[anchor]), mount_candidates=candidates,
                  mount_selected_start=selected["start"], bottle_to_simulation=transform.tolist(),
                  phase_metrics=phase_metrics, joint_near_limit_fraction=near_limit.mean(0).tolist(),
                  max_control_step_rad=float(np.abs(np.diff(controls, axis=0)).max()),
                  wrist_range_rad=model.actuator_ctrlrange[0].tolist(),
                  source_angle_range_degrees=np.rad2deg([angles.min(), angles.max()]).tolist(),
                  reference_sha256=digest(output / "kinematic_reference.npz"),
                  limitations=["Mount fitted at first annotated unscrew frame, then held fixed for every frame",
                               "No independent per-finger translations or object resizing",
                               "Human fingertip landmarks and robot markers are not exact pad-surface correspondences",
                               "No collision terms yet; good endpoint error does not prove a physically valid grip",
                               "Late human hand transport may be unreachable for the fixed-base robot",
                               "This is direct landmark IK, not DexMV policy training or a successful residual controller"])
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(phase_metrics=phase_metrics, reference=report["reference_sha256"])), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=Path("artifacts/orca_formal/CAP03_derived_joint_validation/human_cap.npz"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/orca_collision_refit/O20_hic53_elliptic/cube.xml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("stride must be positive")
    fit(args.packet, args.model, args.output, args.stride)


if __name__ == "__main__":
    main()
