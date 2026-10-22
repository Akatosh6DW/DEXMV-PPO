"""Reconstruct showcase contact locations without advancing or changing policies."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def audit(case, stride):
    model = mujoco.MjModel.from_xml_path(str(ROOT / case["model"]))
    data = mujoco.MjData(model)
    cube = model.geom("cube").id
    with np.load(ROOT / case["source_trace"], allow_pickle=False) as archive:
        trace = {key: archive[key][case["episode"]] for key in archive.files}
    rows = []
    frames = 0
    for i in range(59, min(case["endpoint"] + 1, len(trace["qpos"])), stride):
        if not trace["active"][i]:
            continue
        data.qpos[:] = trace["qpos"][i]
        data.qvel[:] = trace["qvel"][i]
        data.ctrl[:] = trace["ctrl"][i]
        data.mocap_pos[:] = trace["mocap"][i]
        mujoco.mj_forward(model, data)
        frames += 1
        for c in data.contact[:data.ncon]:
            if cube not in (c.geom1, c.geom2) or c.dist > 0:
                continue
            gid = c.geom2 if c.geom1 == cube else c.geom1
            name = model.geom(gid).name
            if not name.startswith("orca_"):
                continue
            body = model.geom_bodyid[gid]
            rotation = data.xmat[body].reshape(3, 3)
            point = (c.pos - data.xpos[body]) @ rotation
            # MuJoCo's contact normal points from geom1 toward geom2.
            normal = c.frame[:3] * (1 if c.geom1 == gid else -1)
            local_normal = normal @ rotation
            rows.append(dict(case=case["id"], frame=i, time=(i + 1) * .05,
                             geom=name, body=model.body(body).name,
                             x=float(point[0]), y=float(point[1]), z=float(point[2]),
                             nx=float(local_normal[0]), ny=float(local_normal[1]),
                             nz=float(local_normal[2]), penetration=-float(c.dist)))
    fingers = {}
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        selected = [r for r in rows if r["geom"].startswith("orca_" + finger + "_")]
        distal = [r for r in selected if "_tip_" in r["geom"]]
        fingers[finger] = dict(
            contacts=len(selected), geoms=dict(Counter(r["geom"] for r in selected)),
            distal_contacts=len(distal),
            distal_lateral_fraction=(float(np.mean([abs(r["ny"]) > .7071 for r in distal]))
                                     if distal else None),
            distal_minus_x_fraction=(float(np.mean([-r["nx"] > .5 for r in distal]))
                                     if distal else None),
            distal_mean_normal=(np.mean([[r[k] for k in ("nx", "ny", "nz")] for r in distal], 0).tolist()
                                if distal else None),
            distal_mean_z=(float(np.mean([r["z"] for r in distal])) if distal else None))
    return rows, dict(case=case["id"], sampled_frames=frames, fingers=fingers,
                     source_trace=case["source_trace"],
                     trace_sha256=hashlib.sha256((ROOT / case["source_trace"]).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "artifacts/presentation_clips_20261012/plan.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/orca_benchmark_suite/S05/diagnosis")
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, reports = [], []
    for case in json.loads(args.plan.read_text()):
        contacts, report = audit(case, args.stride)
        rows.extend(contacts)
        reports.append(report)
        print(case["id"], json.dumps(report["fingers"]["index"]), flush=True)
    with (args.output / "contacts.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "report.json").write_text(json.dumps(dict(
        limitation="Saved-state reconstruction, not original substep or force replay; showcase cases only.",
        axes="Body-local coordinates; lateral = abs(normal_y)>cos(45deg). -X interpretation needs CAD check.",
        cases=reports), indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
