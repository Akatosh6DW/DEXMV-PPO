"""Rigidly align an S05 hand/object pair to a flat resting object, then probe."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from probe_hic53_orca import native_rollout
from retarget_s05_pads import MODEL
from s05_pad_contacts import pad_contacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=3.)
    parser.add_argument("--seconds", type=float, default=8.)
    parser.add_argument("--up-axis", choices=("y", "negative-y"), default="y")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.input, allow_pickle=False) as archive:
        packet = {k: archive[k].copy() for k in archive.files}
    target = np.array([[-1., 0, 0], [0, 0, 1.], [0, 1., 0]])
    if args.up_axis == "negative-y":
        target = np.array([[1., 0, 0], [0, 0, 1.], [0, -1., 0]])
    change = target @ packet["object_to_world_rotation"].T
    new_position = np.array([.055, .025, .265])
    shift = new_position - change @ packet["object_position"]
    root = ET.parse(MODEL).getroot()
    mount = root.find(".//body[@name='right_mount']")
    old_pos = np.fromstring(mount.get("pos"), sep=" ")
    old_quat = np.fromstring(mount.get("quat"), sep=" ")
    rotation = change @ Rotation.from_quat(old_quat[[1, 2, 3, 0]]).as_matrix()
    quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    mount.set("pos", " ".join(map(str, change @ old_pos + shift)))
    mount.set("quat", " ".join(map(str, quat)))
    packet["object_position"] = new_position
    packet["object_to_world_rotation"] = target
    packet["object_quaternion"] = Rotation.from_matrix(target).as_quat()[[3, 0, 1, 2]]
    old_ctrl, old_time = packet["ctrl"], packet["time"]
    times = np.arange(round(args.duration / .05) + 1) * .05
    ctrl = np.stack([np.interp(times / args.duration, old_time / old_time[-1], old_ctrl[:, k])
                     for k in range(17)], axis=1)
    packet.update(ctrl=ctrl, time=times, mount_world_transform=change, mount_world_translation=shift)
    for name, pos in (("cube", new_position), ("cube_support", new_position - [0, 0, .0305])):
        root.find(f".//body[@name='{name}']").set("pos", " ".join(map(str, pos)))
    root.find(".//body[@name='cube']").set("quat", " ".join(map(str, packet["object_quaternion"])))
    ET.ElementTree(root).write(args.output / "model.xml", encoding="unicode")
    np.savez_compressed(args.output / "reference.npz", **packet)
    model = mujoco.MjModel.from_xml_path(str(args.output / "model.xml"))
    reports = []
    for seed in range(5001, 5005):
        trace, report = native_rollout(model, new_position, packet["object_quaternion"], ctrl, seed,
                                       seconds=args.seconds, initial_ctrl=packet["initial_ctrl"])
        data = mujoco.MjData(model)
        valid = []
        for i in range(len(trace["qpos"])):
            data.qpos[:], data.qvel[:], data.ctrl[:] = trace["qpos"][i], trace["qvel"][i], trace["ctrl"][i]
            data.mocap_pos[:] = trace["mocap"][i]
            mujoco.mj_forward(model, data)
            good, bad, _ = pad_contacts(model, data)
            valid.append(bool((good > .02).all() and not (bad > .02).any() and trace["hold"][i] > 0))
        best = count = 0
        for ok in valid:
            count = count + 1 if ok else 0
            best = max(count, best)
        report["reconstructed_pad_hold_sec"] = best * .05
        report["reconstructed_pad_valid_fraction"] = float(np.mean(valid[59:])) if len(valid) > 59 else 0.
        reports.append(report)
        np.savez_compressed(args.output / f"probe_{seed}.npz", **trace, pad_valid=valid)
        print(json.dumps(report), flush=True)
    (args.output / "report.json").write_text(json.dumps(dict(
        input=str(args.input), duration=args.duration, seconds=args.seconds,
        up_axis=args.up_axis, probes=reports,
        change="Rigid hand mount and object alignment only; no collision/inertia/friction edits",
        warning="Pad metrics reconstruct saved control frames, not a full substep validation",
        rotation=change.tolist(), translation=shift.tolist()), indent=2) + "\n")


if __name__ == "__main__":
    main()
