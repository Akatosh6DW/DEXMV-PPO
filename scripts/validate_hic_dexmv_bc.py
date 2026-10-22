#!/usr/bin/env python3
"""Check DexMV BC pose tracking and free-cube grasp in native MuJoCo 2.1."""

import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco_py
import numpy as np
from scipy.spatial.transform import Rotation


def xyz(value):
    return " ".join(map(str, value))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("artifacts/cube20/hic20"))
    p.add_argument("--frame", type=int, default=95)
    p.add_argument("--seconds", type=float, default=6)
    p.add_argument("--retarget-dir", default="dexmv_retarget")
    p.add_argument("--bc-dir", default="dexmv_bc")
    a = p.parse_args()
    ref = np.load(a.root / a.retarget_dir / "adroit_cube.npz")
    learned = np.load(a.root / a.bc_dir / "learned_prior.npz")
    idx = int(np.argmin(np.abs(learned["time"] - ref["time"][a.frame])))
    target = learned["adroit_qpos"][idx]
    obj = ref["object_poses"][a.frame]
    xml = ET.fromstring((a.root / a.retarget_dir / "adroit_hic.xml").read_text())
    xml.find("option").set("timestep", "0.002")
    world = xml.find("worldbody")
    ET.SubElement(world, "light", pos="0 -0.1 1", diffuse="0.8 0.8 0.8")
    body = ET.SubElement(world, "body", name="cube", pos=xyz(obj[:3, 3]))
    ET.SubElement(body, "freejoint", name="cube_free")
    ET.SubElement(body, "geom", name="cube_geom", type="box", size="0.0265 0.0265 0.0265", mass="0.075", rgba="0.2 0.6 0.9 1", friction="1 0.005 0.0001", condim="4")
    bottom = obj[2, 3] - np.sum(np.abs(obj[2, :3])) * .0265
    support = ET.SubElement(world, "body", name="cube_support", mocap="true", pos=xyz([obj[0, 3], obj[1, 3], bottom - .01]))
    ET.SubElement(support, "geom", name="support_geom", type="box", size="0.04 0.04 0.01", rgba="0.5 0.5 0.5 1")
    out = a.root / a.bc_dir / "dynamic_check"
    out.mkdir(parents=True, exist_ok=True)
    xml_text = ET.tostring(xml, encoding="unicode")
    (out / "scene.xml").write_text(xml_text)
    model = mujoco_py.load_model_from_xml(xml_text)
    sim = mujoco_py.MjSim(model)
    sim.data.qpos[:30] = target
    sim.data.qpos[30:33] = obj[:3, 3]
    quat = Rotation.from_matrix(obj[:3, :3]).as_quat()
    sim.data.qpos[33:37] = quat[[3, 0, 1, 2]]
    sim.forward()
    cube_gid = model.geom_name2id("cube_geom")
    support_gid = model.geom_name2id("support_geom")
    rows, run, best = [], 0, 0
    for step in range(round(a.seconds / model.opt.timestep)):
        if sim.data.time >= 1.0:
            sim.data.mocap_pos[0, 2] = -2.0
        # Adroit general actuators are affine: kp and bias slopes differ.
        # Invert that law instead of incorrectly treating qpos as raw ctrl.
        ctrl = (sim.data.qfrc_bias[:30] - model.actuator_biasprm[:, 1] * target) / model.actuator_gainprm[:, 0]
        sim.data.ctrl[:] = np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        sim.step()
        fingers, supported = set(), False
        for c in sim.data.contact[:sim.data.ncon]:
            if cube_gid not in [c.geom1, c.geom2] or c.dist > 0:
                continue
            other = c.geom2 if c.geom1 == cube_gid else c.geom1
            supported |= other == support_gid
            name = model.geom_id2name(other) or ""
            for prefix in ["th", "ff", "mf", "rf", "lf"]:
                if name.startswith("C_" + prefix):
                    fingers.add(prefix)
        hold = sim.data.time >= 1 and not supported and len(fingers) >= 2 and sim.data.qpos[32] > obj[2, 3] - .02
        run = run + 1 if hold else 0
        best = max(best, run)
        if step % 5 == 0:
            rows.append({"time": float(sim.data.time), "finger_tracking_rmse_rad": float(np.sqrt(np.mean((sim.data.qpos[8:30] - target[8:30]) ** 2))),
                         "cube_z": float(sim.data.qpos[32]), "finger_contacts": len(fingers), "support_contact": supported,
                         "continuous_hold_sec": run * model.opt.timestep})
    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report = {"start_video_frame": a.frame, "seconds": a.seconds,
              "max_unsupported_hold_sec": best * model.opt.timestep,
              "first": rows[0], "last": rows[-1],
              "note": "BC reference held at grasp phase; free cube, gravity, support withdrawn after 1s; no RL updates"}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
