"""Object-relative, bounded seven-joint IK for the unchanged S02 Orca task."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from diagnose_orca_grasp import model_indices
from orca_cylinder_s02_env import cylinder_xml, SOURCE_REFERENCE
from prepare_cylinder_s02_source import digest
from probe_hic53_orca import cube_penetration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.segment, allow_pickle=False) as z:
        source = {k: z[k].copy() for k in z.files}
    with np.load(SOURCE_REFERENCE, allow_pickle=False) as z:
        metadata = {k: z[k].copy() for k in ("initial_ctrl", "object_position", "object_quaternion",
                     "object_to_world_rotation", "object_extents_m", "spatial_scale")}
    xml = cylinder_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    qids, object_qid, tip_ids = model_indices(model)
    tips = tip_ids[:2]
    pads = np.array([model.geom('orca_thumb_tip_proxy_fit_02').id,
                     model.geom('orca_index_tip_proxy_fit_02').id])
    chosen = np.arange(10, 17)
    initial = metadata["initial_ctrl"]
    position, world_rotation = metadata["object_position"], metadata["object_to_world_rotation"]
    data.qpos[object_qid:object_qid+7] = np.r_[position, metadata["object_quaternion"]]
    low, high = model.actuator_ctrlrange[chosen].T
    low, high = low+.08, high-.08
    # Dimension normalization is explicit: a source 70x104 mm can becomes the
    # 53x53 mm task object. Only object-relative geometry is scaled, not physics.
    source_points = source["fingertips_centered"][:, :2]
    scale = .053/source["source_extents_m"]
    raw = source_points*scale
    corrected = raw.copy()
    radial = np.linalg.norm(source_points[:, :, :2], axis=-1)
    source_radius = source["source_extents_m"][0]/2
    normal = source_points[:, :, :2]/np.maximum(radial[:, :, None], 1e-9)
    near = np.clip((.025-np.abs(radial-source_radius))/.010, 0, 1)
    # *_tip_ball_proxy is a non-colliding 0.1 mm marker, not a fingertip.
    # Match the distal collision capsules' centers to offset surface targets.
    radii = model.geom_size[pads, 0]
    target_radius = .0265+radii-.0005
    corrected[:, :, :2] += near[:, :, None]*(normal*target_radius[None, :, None]-corrected[:, :, :2])
    source_to_target = Rotation.from_euler("x", np.pi/2).as_matrix()
    cube_gid = model.geom("cube").id

    def place(ctrl):
        q = initial.copy()
        q[chosen] = ctrl
        data.qpos[qids] = q
        mujoco.mj_forward(model, data)

    def penalty():
        result = np.zeros(model.ngeom)
        for contact in data.contact[:data.ncon]:
            if cube_gid in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == cube_gid else contact.geom1
                if model.geom_contype[other] == 1:
                    result[other] = max(result[other], -float(contact.dist)-.0005)
        return 5*result

    def targets(points, angle):
        rotation = Rotation.from_euler("y", angle).as_matrix() @ source_to_target
        return points @ rotation.T @ world_rotation.T+position

    def endpoint(x):
        place(x[:7])
        desired = targets(corrected[-1], x[7])
        return np.r_[(data.geom_xpos[pads]-desired).ravel(), .001*x[:7], penalty()]

    rng = np.random.default_rng(101)
    candidates = []
    for i in range(12):
        start = np.r_[np.clip(initial[chosen], low+1e-8, high-1e-8) if i == 0 else rng.uniform(low, high),
                       0. if i == 0 else rng.uniform(-np.pi, np.pi)]
        fit = least_squares(endpoint, start, bounds=(np.r_[low, -np.pi], np.r_[high, np.pi]),
                            max_nfev=250, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        residual = endpoint(fit.x)
        candidates.append(dict(start=i, x=fit.x.tolist(), marker_error_m=float(np.linalg.norm(residual[:6])),
                               penetration_m=cube_penetration(model, data)))
    best = min(candidates, key=lambda c: c["marker_error_m"]+5*max(c["penetration_m"]-.001, 0))
    angle = best["x"][7]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"model.xml").write_text(xml)
    results = {}
    for name, points in (("unrefined", raw), ("reference", corrected)):
        tracking_geoms = tips if name == 'unrefined' else pads
        desired = targets(points, angle)
        previous = np.array(best["x"][:7])
        controls = np.tile(initial, (len(points), 1))
        errors = np.zeros((len(points), 2))
        for frame in reversed(range(len(points))):
            def residual(x):
                place(x)
                return np.r_[(data.geom_xpos[tracking_geoms]-desired[frame]).ravel(),
                             .002*(x-previous), penalty()]
            fit = least_squares(residual, previous, bounds=(low, high), max_nfev=150,
                                ftol=1e-8, xtol=1e-8, gtol=1e-8)
            residual(fit.x)
            errors[frame] = np.linalg.norm(data.geom_xpos[tracking_geoms]-desired[frame], axis=1)
            controls[frame, chosen], previous = fit.x, fit.x
        times = np.arange(31)*.05
        fraction = times/1.5
        ctrl = np.stack([np.interp(fraction, source["time"]/source["time"][-1], controls[:, k])
                         for k in range(17)], axis=1)
        ctrl = initial+fraction[:, None]*(ctrl-initial)
        np.savez_compressed(args.output/(name+".npz"), ctrl=ctrl, time=times, **metadata,
                            raw_ik_ctrl=controls, source_local_tips=source_points, target_tips=desired,
                            errors_m=errors, source_dimension_scale=scale, source_sequence=source["source_sequence"],
                            source_indices=source["source_indices"], source_time=source["time"],
                            cylinder_axis_in_task_object_frame=[0, 1, 0])
        results[name] = dict(mean_marker_error_mm=(errors.mean(0)*1000).tolist(),
                             endpoint_error_mm=(errors[-1]*1000).tolist(), sha256=digest(args.output/(name+".npz")))
    report = dict(source_segment_sha256=digest(args.segment), source_sequence=int(source["source_sequence"]),
                  original_task_metadata_sha256=digest(SOURCE_REFERENCE), task_pose_and_initial_ctrl_unchanged=True,
                  source_dimension_scale=scale.tolist(), azimuth_rad=angle, candidates=candidates, results=results,
                  selected_actuators=chosen.tolist(), joint_margin_rad=.08, collision_weight=5.,
                  alignment="Shared cylinder-axis/azimuth transform; no independent fingertip translation",
                  correction="Distal physical capsule center/radius surface matching; no rollout/reward search",
                  contact_geoms=[model.geom(int(i)).name for i in pads],
                  contact_radii_m=radii.tolist(),
                  raw_control_definition="Unrefined uses scaled published marker centers; refined matches contact surfaces",
                  model_sha256=digest(args.output/"model.xml"), trainable=False)
    (args.output/"retarget_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(selected=best, results=results)), flush=True)


if __name__ == "__main__":
    main()
