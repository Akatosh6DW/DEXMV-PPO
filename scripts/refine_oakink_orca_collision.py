"""Add explicit hand/object nonpenetration to the fixed-bottle landmark fit."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
import trimesh

from prepare_hic_orca_object import digest
from probe_oakink_orca_retarget import FINGERS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collision-weight", type=float, default=10.)
    parser.add_argument("--surface-targets", action="store_true")
    parser.add_argument("--anchor-first", action="store_true")
    parser.add_argument("--mount-frames", type=int, default=1)
    args = parser.parse_args()
    if not np.isfinite(args.collision_weight) or args.collision_weight <= 0:
        parser.error("collision-weight must be finite and positive")
    if args.mount_frames < 1:
        parser.error("mount-frames must be positive")
    with np.load(args.source) as source:
        z = {k: source[k] for k in source.files}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "refinement_source.py").write_bytes(Path(__file__).read_bytes())
    tree = ET.parse(args.scene)
    tree.find(".//body[@name='oakink_bottle']").set("mocap", "true")
    for mesh in tree.findall("asset/mesh"):
        path = Path(mesh.get("file"))
        if not path.is_absolute():
            mesh.set("file", str((args.scene.parent / path).resolve()))
    tree.write(args.output / "optimizer_scene.xml", encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(args.output / "optimizer_scene.xml"))
    data = mujoco.MjData(model)
    qids = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    tips = [model.geom(f"orca_{f}_tip_ball_proxy").id for f in FINGERS]
    object_ids = [model.geom(f"{name}_collision").id for name in ("bottle", "cap")]
    mocaps = [model.body(f"oakink_{name}").mocapid[0] for name in ("bottle", "cap")]
    low, high = model.actuator_ctrlrange.T
    weights = np.array([1., 1., .15, .15, .15])[:, None]
    original_transform = z["bottle_to_simulation"]
    local_targets = (z["target_tips"] - original_transform[:3, 3]) @ original_transform[:3, :3]
    contact_hypothesis = np.zeros((len(local_targets), 2), dtype=bool)
    projection_shift = np.zeros((len(local_targets), 2))
    if args.surface_targets:
        # A video landmark is a bone endpoint, not the robot pad's centre.
        # Only thumb/index proximity is used here; this is an explicitly
        # labelled contact hypothesis, not measured source contact force.
        mesh = trimesh.load(args.scene.parent / "cap_convex.stl", force="mesh", process=True)
        for finger, suffix in enumerate(("02", "03")):
            tips[finger] = model.geom(f"orca_{FINGERS[finger]}_tip_proxy_fit_{suffix}").id
        for frame, cap in enumerate(z["source_cap_in_bottle"]):
            points = (local_targets[frame, :2] - cap[:3, 3]) @ cap[:3, :3]
            surface, distance, triangles = trimesh.proximity.closest_point_naive(mesh, points)
            normal = mesh.face_normals[triangles]
            for finger in range(2):
                if distance[finger] <= .025:
                    centre = surface[finger] + model.geom_size[tips[finger], 0] * normal[finger]
                    target = cap[:3, :3] @ centre + cap[:3, 3]
                    projection_shift[frame, finger] = np.linalg.norm(target - local_targets[frame, finger])
                    local_targets[frame, finger] = target
                    contact_hypothesis[frame, finger] = True
    anchor = int(np.argmin(abs(z["source_frame_id"] - 423)))
    initial = z["ctrl"][anchor].copy()

    def decode(x):
        transform = original_transform.copy()
        transform[:3, :3] = Rotation.from_rotvec(x[17:20]).as_matrix() @ original_transform[:3, :3]
        transform[:3, 3] += x[20:23]
        return transform

    def geometry(ctrl, frame, transform):
        data.qpos[qids] = ctrl
        for mocap, pose in zip(mocaps, (transform, transform @ z["source_cap_in_bottle"][frame])):
            data.mocap_pos[mocap] = pose[:3, 3]
            data.mocap_quat[mocap] = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        depths = np.zeros((2, model.ngeom))
        for contact in data.contact[:data.ncon]:
            for i, gid in enumerate(object_ids):
                if gid not in (contact.geom1, contact.geom2):
                    continue
                other = contact.geom2 if contact.geom1 == gid else contact.geom1
                if model.geom_contype[other] == 1:
                    depths[i, other] = max(depths[i, other], -float(contact.dist))
        return data.geom_xpos[tips].copy(), depths

    def mount_loss(x):
        transform = decode(x)
        target = local_targets[anchor] @ transform[:3, :3].T + transform[:3, 3]
        fitted, depths = geometry(x[:17], anchor, transform)
        return np.r_[((fitted - target) * weights).ravel(),
                     args.collision_weight * np.maximum(depths - .0003, 0).ravel(),
                     .001 * (x[:17] - initial), .004 * x[17:20], .01 * x[20:23]]

    if args.mount_frames == 1:
        mount = least_squares(mount_loss, np.r_[initial, np.zeros(6)],
                              bounds=(np.r_[low, [-.8] * 3, [-.04] * 3],
                                      np.r_[high, [.8] * 3, [.04] * 3]), max_nfev=350,
                              ftol=1e-8, xtol=1e-8, gtol=1e-8)
        mount_frames = [anchor]
    else:
        last = int(np.argmin(abs(z["source_frame_id"] - 1500)))
        mount_frames = np.unique(np.linspace(anchor, last, args.mount_frames, dtype=int)).tolist()
        n = len(mount_frames)
        block = 15 + 2 * model.ngeom + 17

        def batch_loss(x):
            transform = decode(np.r_[np.zeros(17), x[-6:]])
            controls = x[:-6].reshape(n, 17)
            parts = []
            for ctrl, frame in zip(controls, mount_frames):
                target = local_targets[frame] @ transform[:3, :3].T + transform[:3, 3]
                fitted, depths = geometry(ctrl, frame, transform)
                parts.append(np.r_[((fitted - target) * weights).ravel(),
                                   args.collision_weight * np.maximum(depths - .0003, 0).ravel(),
                                   .001 * (ctrl - z["ctrl"][frame])])
            return np.r_[np.concatenate(parts), .004 * x[-6:-3], .01 * x[-3:]]

        sparse = lil_matrix((block * n + 6, 17 * n + 6), dtype=int)
        for i in range(n):
            sparse[block * i:block * (i + 1), 17 * i:17 * (i + 1)] = 1
            sparse[block * i:block * (i + 1), -6:] = 1
        sparse[-6:, -6:] = 1
        mount = least_squares(batch_loss, np.r_[z["ctrl"][mount_frames].ravel(), np.zeros(6)],
                              bounds=(np.r_[np.tile(low, n), [-.8] * 3, [-.04] * 3],
                                      np.r_[np.tile(high, n), [.8] * 3, [.04] * 3]),
                              jac_sparsity=sparse.tocsr(), max_nfev=350,
                              ftol=1e-7, xtol=1e-7, gtol=1e-7)
        mount.x = np.r_[mount.x[:17], mount.x[-6:]]
    transform = decode(mount.x)
    targets = local_targets @ transform[:3, :3].T + transform[:3, 3]
    print(json.dumps(dict(stage="mount_fit", evaluations=mount.nfev, objective=float(np.square(mount.fun).sum()))), flush=True)
    previous = mount.x[:17].copy()
    controls, errors, fitted_points, depth_trace = ([None] * len(targets) for _ in range(4))
    # Approach/transport remain in the packet for honest reachability reporting;
    # no claim is made that this fixed-base robot can reproduce arm translation.
    order = (list(range(anchor, len(targets))) + list(reversed(range(anchor)))
             if args.anchor_first else range(len(targets)))
    for frame in order:
        if args.anchor_first and frame == anchor - 1:
            previous = mount.x[:17].copy()
        def residual(ctrl):
            fitted, depths = geometry(ctrl, frame, transform)
            return np.r_[((fitted - targets[frame]) * weights).ravel(),
                         args.collision_weight * np.maximum(depths - .0003, 0).ravel(),
                         .002 * (ctrl - previous)]

        result = least_squares(residual, np.clip(previous, low + 1e-8, high - 1e-8), bounds=(low, high),
                               max_nfev=100, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        fitted, depths = geometry(result.x, frame, transform)
        controls[frame] = result.x.copy()
        errors[frame] = np.linalg.norm(fitted - targets[frame], axis=1)
        fitted_points[frame] = fitted
        depth_trace[frame] = depths.max(axis=1)
        previous = result.x.copy()
        if frame % 40 == 0:
            print(json.dumps(dict(frame=int(z["source_frame_id"][frame]), errors_mm=(1000 * errors[frame][:2]).tolist(),
                                  bottle_cap_depth_mm=(1000 * depths.max(axis=1)).tolist())), flush=True)
    errors, depths = np.asarray(errors), np.asarray(depth_trace)
    z.update(ctrl=np.asarray(controls), target_tips=targets, fitted_tips=np.asarray(fitted_points), errors_m=errors,
             bottle_to_simulation=transform, cap_to_simulation=transform @ z["source_cap_in_bottle"],
             bottle_cap_depth_m=depths, contact_hypothesis=contact_hypothesis,
             projection_shift_m=projection_shift, target_geom_names=np.asarray([model.geom(i).name for i in tips]))
    z.pop("base_errors_m", None)
    np.savez_compressed(args.output / "kinematic_reference.npz", **z)
    phase = (z["source_frame_id"] >= 423) & (z["source_frame_id"] < 1548)
    report = dict(kind="collision_aware_landmark_IK_not_PPO_not_dynamic_success", source_sha256=digest(args.source),
                  scene_sha256=digest(args.scene), script_sha256=digest(Path(__file__)),
                  collision_weight=args.collision_weight, allowed_penetration_m=.0003,
                  surface_targets=args.surface_targets,
                  anchor_first=args.anchor_first,
                  mount_source_frames=z["source_frame_id"][mount_frames].tolist(),
                  contact_hypothesis="Thumb/index within 25 mm of source hull; target pad centre one actual pad radius outside nearest surface" if args.surface_targets else None,
                  mean_projection_shift_mm=(1000 * projection_shift[phase].mean(0)).tolist(),
                  target_geom_names=[model.geom(i).name for i in tips],
                  mount_fit_converged=bool(mount.success), mount_parameters=mount.x.tolist(),
                  bottle_to_simulation=transform.tolist(),
                  phase="source IDs [423,1548), before removal transport",
                  mean_tip_error_mm=(1000 * errors[phase].mean(0)).tolist(),
                  p95_tip_error_mm=(1000 * np.percentile(errors[phase], 95, axis=0)).tolist(),
                  max_bottle_cap_depth_mm=(1000 * depths[phase].max(0)).tolist(),
                  fraction_bottle_cap_over_3mm=(depths[phase] > .003).mean(0).tolist(),
                  contact_pair_under_10mm_fraction=float((errors[phase, :2].max(1) <= .01).mean()),
                  reference_sha256=digest(args.output / "kinematic_reference.npz"))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
