"""Source-relative three-tip errors for references and actual trained rollouts."""

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from diagnose_orca_grasp import model_indices
from orca_s06_env import PAD_NAMES, cylinder_contacts

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((a.run / "run_config.json").read_text())
    model = mujoco.MjModel.from_xml_path(config["model"])
    data = mujoco.MjData(model)
    pads = [model.geom(n).id for n in PAD_NAMES]
    bodies = model.geom_bodyid[pads]
    oid = model.body("cube").id
    qids, oqid, _ = model_indices(model)
    with np.load(config["reference"]) as z:
        packet = dict(z)
    target = packet["target_body_tips"][-1]

    def measure():
        rot = data.xmat[oid].reshape(3, 3)
        actual = (data.geom_xpos[pads]-data.xpos[oid]) @ rot
        error = np.linalg.norm(actual-target, axis=1)*1000
        inward = -actual.copy()
        inward[:, 1] = 0
        inward /= np.maximum(np.linalg.norm(inward, axis=1)[:, None], 1e-9)
        normal = -data.xmat[bodies].reshape(3, 3, 3)[:, :, 0] @ rot
        return error, np.sum(normal*inward, axis=1)

    data.qpos[qids] = packet["ctrl"][-1]
    data.qpos[oqid:oqid+7] = np.r_[packet["object_position"], packet["object_quaternion"]]
    data.mocap_pos[0] = [0, 0, -1]
    mujoco.mj_forward(model, data)
    static_error, static_cos = measure()
    old_dir = ROOT / "artifacts/orca_cylinder_s02/inputs"
    with np.load(old_dir / "reference.npz") as z:
        old = dict(z)
    report_file = old_dir / "retarget_report.json"
    if not report_file.exists():
        report_file = ROOT / "artifacts/orca_cylinder_s02/inputs03/retarget_report.json"
    angle = json.loads(report_file.read_text())["azimuth_rad"]
    new_target = target.copy()
    target = target @ Rotation.from_euler("y", angle).as_matrix().T
    data.qpos[qids] = old["ctrl"][-1]
    data.qpos[oqid:oqid+7] = np.r_[old["object_position"], old["object_quaternion"]]
    mujoco.mj_forward(model, data)
    old_error, old_cos = measure()
    target = new_target
    directory = a.run / f"step_{a.step:09d}"
    rows, control_errors = [], []
    with np.load(directory / "preview_trajectories.npz") as z:
        trace = dict(z)
    for episode in range(len(trace["qpos"])):
        for frame in range(59, 800, 2):
            if not trace["active"][episode, frame]:
                continue
            for field, key in (("qpos", "qpos"), ("qvel", "qvel"), ("ctrl", "ctrl"), ("mocap_pos", "mocap")):
                getattr(data, field)[:] = trace[key][episode, frame]
            mujoco.mj_forward(model, data)
            error, cosine = measure()
            good, bad, _ = cylinder_contacts(model, data)
            row = dict(episode=episode, time=(frame+1)*.05, valid=bool(trace["hold"][episode, frame] > 0))
            for i, finger in enumerate(("thumb", "index", "middle")):
                row.update({finger+"_error_mm": float(error[i]), finger+"_facing_cos": float(cosine[i]),
                            finger+"_good_n": float(good[i]), finger+"_bad_n": float(bad[i])})
            rows.append(row)
        if config["method"] == "residual":
            active = trace["active"][episode]
            previous = np.vstack([packet["initial_ctrl"], trace["ctrl"][episode, :-1]])
            requested = .08*np.clip(trace["action"][episode], -1, 1)
            desired = np.clip(trace["video_base"][episode]+requested, *model.actuator_ctrlrange.T)
            actual = np.clip(desired, previous-.1, previous+.1)
            control_errors.append(float(np.max(np.abs(actual[active]-trace["ctrl"][episode, active]))))
    with (a.output / "contacts_and_similarity.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    valid_rows = [r for r in rows if r["valid"]]
    summary = dict(run=str(a.run), fingers=["thumb", "index", "middle"],
        metric="Distance between distal proxy centers and fixed source-endpoint surface targets in actual object frame; shared cylinder-axis alignment, no per-finger fitting",
        old_reference_errors_mm=old_error.tolist(), new_reference_errors_mm=static_error.tolist(),
        old_facing_cos=old_cos.tolist(), new_facing_cos=static_cos.tolist(),
        evaluated_saved_episodes=len(trace["qpos"]), sampled_contact_rows=len(rows), valid_rows=len(valid_rows),
        valid_state_mean_errors_mm=[float(np.mean([r[f+"_error_mm"] for r in valid_rows])) if valid_rows else None for f in ("thumb", "index", "middle")],
        residual_control_max_error_rad=max(control_errors, default=None),
        limitation="Not full human pose matching; source has reconstruction errors, and old/new task placements and solver settings differ. Contacts reconstructed every100ms; runtime5ms audit determines success.")
    (a.output / "report.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
