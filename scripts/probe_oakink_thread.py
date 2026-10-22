"""Native mechanism controls and video-only replay, never trained-policy scores."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from oakink_cap_dynamics import apply_thread_contact_mode, contact_report, step_passive
from prepare_hic_orca_object import digest


def probe(model_path, meta, reference, mode, seconds=20.):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    isolated = mode in ("isolated_zero", "isolated_torque")
    if isolated:
        hand = model.geom_contype == 1
        model.geom_contype[hand] = 0
        model.geom_conaffinity[hand] = 0
    data = mujoco.MjData(model)
    qids = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    turn, lift = [model.joint(name).qposadr[0] for name in ("cap_turn", "cap_lift")]
    turn_dof = model.joint("cap_turn").dofadr[0]
    pitch, release = [meta["parameters"][k] for k in ("pitch_m_per_rad", "release_angle_rad")]
    initial = np.asarray(meta["initial_ctrl"])
    if mode == "open_hand":
        initial = np.zeros(17)
        initial[0] = -.25
    data.qpos[qids], data.ctrl[:] = initial, initial
    apply_thread_contact_mode(model, data)
    mujoco.mj_forward(model, data)
    rows = []
    anchor = int(np.argmin(abs(reference["source_frame_id"] - meta["source_anchor_frame"])))
    times = reference["time"][anchor:] - reference["time"][anchor]
    initial_contacts = contact_report(model, data)
    for step in range(round(seconds / .05)):
        if mode == "video_only":
            desired = np.array([np.interp((step + 1) * .05, times, reference["ctrl"][anchor:, j]) for j in range(17)])
        else:
            desired = initial
        data.ctrl[:] = np.clip(desired, data.ctrl - .1, data.ctrl + .1)
        torque = .005 if mode == "isolated_torque" and data.eq_active.any() else 0.
        data.qfrc_applied[turn_dof] = torque
        step_passive(model, data, release, round(.05 / model.opt.timestep))
        depths = contact_report(model, data)
        force = np.zeros(6)
        peak = 0.
        for i in range(data.ncon):
            mujoco.mj_contactForce(model, data, i, force)
            peak = max(peak, float(force[0]))
        rows.append(dict(qpos=data.qpos.copy(), qvel=data.qvel.copy(), ctrl=data.ctrl.copy(),
                         angle=float(data.qpos[turn]), lift=float(data.qpos[lift]),
                         thread_active=bool(data.eq_active.any()),
                         thread_error_m=float(data.qpos[lift] - pitch * data.qpos[turn]),
                         imposed_test_torque_nm=torque, peak_force_n=peak, **depths))
    trace = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    engaged = trace["thread_active"]
    result = dict(mode=mode, hand_collisions_disabled_for_isolated_mechanism_control=isolated,
                  reset_valid_for_hand_contacts=max(initial_contacts["hand_cap_depth_m"], initial_contacts["hand_bottle_depth_m"]) <= .003,
                  final_angle_degrees=float(np.rad2deg(trace["angle"][-1])),
                  max_angle_degrees=float(np.rad2deg(trace["angle"].max())),
                  release_ever=bool((~engaged).any()),
                  release_time_s=float((np.flatnonzero(~engaged)[0] + 1) * .05) if (~engaged).any() else None,
                  max_engaged_thread_error_mm=float(1000 * np.abs(trace["thread_error_m"][engaged]).max()) if engaged.any() else None,
                  peak_force_n=float(trace["peak_force_n"].max()), initial_contacts=initial_contacts,
                  max_depth_mm={k: float(trace[k].max() * 1000) for k in initial_contacts},
                  finite=bool(np.isfinite(trace["qpos"]).all()),
                  learned_policy=False, source_cap_pose_overwritten_during_steps=False)
    return trace, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("artifacts/orca_formal/CAP10_sequence_mount_ik/kinematic_reference.npz"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.model_directory / "report.json").read_text())
    model = args.model_directory / "model.xml"
    if digest(model) != meta["model_sha256"] or digest(args.reference) != meta["reference_sha256"]:
        raise ValueError("Changed model/reference")
    with np.load(args.reference) as packet:
        reference = dict(packet)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "probe_source.py").write_bytes(Path(__file__).read_bytes())
    records = []
    for mode in ("isolated_zero", "isolated_torque", "open_hand", "held_initial", "video_only"):
        trace, result = probe(model, meta, reference, mode)
        np.savez_compressed(args.output / (mode + ".npz"), **trace)
        records.append(result)
        (args.output / "report.json").write_text(json.dumps(dict(stage="mechanism_and_replay_development",
                                                                 model_sha256=digest(model), records=records,
                                                                 complete=len(records) == 5), indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
