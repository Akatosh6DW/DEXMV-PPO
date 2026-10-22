"""Measure each retarget stage in the same object frame, without dynamics."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from diagnose_orca_grasp import model_indices
from prepare_hic_orca_object import digest


def summarize(delta):
    error = np.linalg.norm(delta, axis=-1) * 1000
    return dict(mean_mm=error.mean(axis=0).tolist(),
                p95_mm=np.quantile(error, .95, axis=0).tolist(),
                max_mm=error.max(axis=0).tolist(), endpoint_mm=error[-1].tolist())


def audit(model_path, reference_path, source_path):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    qids, cube_id, tip_ids = model_indices(model)
    with np.load(reference_path) as z:
        ref = {k: z[k] for k in z.files}
    with np.load(source_path) as z:
        source = {k: z[k] for k in z.files}
    start = int(ref["source_frame_start"])
    end = len(ref["raw_ik_ctrl"])
    selection = slice(start, end)
    human = source["human_joints_object_local"][selection][:, [4, 8, 12]]
    target = source["dexmv_targets_object_local"][selection][:, [3, 6, 9]]
    adroit = source["dexmv_fitted_object_local"][selection][:, [3, 6, 9]]
    np.testing.assert_allclose(human, target, atol=1e-6)
    target_source = str(ref.get("target_source", "dexmv-fitted"))
    expected = human if target_source == "human-tracking" else adroit
    np.testing.assert_allclose(expected, ref["source_local_tips"][selection], atol=1e-7)
    np.testing.assert_allclose(source["time"][:end], ref["source_time"], atol=1e-9)
    np.testing.assert_allclose(ref["spatial_scale"], 1., atol=0)
    rotation, position = ref["object_to_world_rotation"], ref["object_position"]
    data.qpos[cube_id:cube_id + 7] = [*position, *ref["object_quaternion"]]
    actual = []
    for ctrl in ref["raw_ik_ctrl"][selection]:
        data.qpos[qids] = ctrl
        mujoco.mj_kinematics(model, data)
        actual.append((data.geom_xpos[tip_ids[:3]] - position) @ rotation)
    actual = np.asarray(actual)
    stages = dict(human_to_adroit=summarize(adroit - human),
                  adroit_to_orca=summarize(actual - adroit),
                  human_to_orca_total=summarize(actual - human))
    report = dict(model_sha256=digest(model_path), reference_sha256=digest(reference_path),
                  source_sha256=digest(source_path), source_frames=[start, end - 1],
                  fingers=["thumb", "index", "middle"], fitted_fingers=["thumb", "index"],
                  middle_is_unconstrained=True, stages=stages, target_source=target_source,
                  note="Marker-center kinematics only, not contact-surface error or dynamic tracking; source RGB fit error is unknown. Total is a vector difference, not sum of stage norms.")
    arrays = dict(source_frame=source["source_video_frame"][selection], human=human,
                  adroit=adroit, orca=actual)
    return report, arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--source", type=Path, default=Path(
        "artifacts/orca_collision_refit/O01_hic53_source_object/source_correspondence.npz"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report, arrays = audit(args.model, args.reference, args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "marker_chain.npz", **arrays)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
