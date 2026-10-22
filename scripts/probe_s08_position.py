"""Position-only longitudinal wrap search with frozen hold criteria and controls."""

import csv
import itertools
import json
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from probe_s08_wrap import BASE, ROOT, rollout, model_indices, cube_penetration
from orca_cylinder_s02_env import cylinder_half_height
from prepare_cylinder_s02_source import digest

SOURCE = ROOT / "artifacts/orca_benchmark_suite/S08_wrap_refined/longitudinal/reference.npz"
OUT = ROOT / "artifacts/orca_benchmark_suite/S08_wrap_position"
DEV = [8301, 8302, 8303, 8304]
TEST = list(range(83200, 83232))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def reset_contacts(m, packet, seed):
    d = mujoco.MjData(m)
    qids, oid, _ = model_indices(m)
    rng = np.random.default_rng(seed)
    spawn = packet["object_position"] + rng.uniform(-.002, .002, 3)*[1, 1, 0]
    d.qpos[qids] = np.clip(packet["initial_ctrl"] + rng.normal(0, .002, 17), *m.actuator_ctrlrange.T)
    d.qpos[oid:oid+7] = np.r_[spawn, packet["object_quaternion"]]
    d.mocap_pos[0] = spawn - [0, 0, cylinder_half_height(packet["object_quaternion"])+.004]
    d.ctrl[:] = packet["initial_ctrl"]
    mujoco.mj_forward(m, d)
    cube = m.geom("cube").id
    pairs = []
    for c in d.contact[:d.ncon]:
        if cube in c.geom and c.dist < 0:
            other = c.geom2 if c.geom1 == cube else c.geom1
            if m.geom_contype[other] == 1:
                pairs.append(dict(geom=m.geom(other).name, depth_m=float(-c.dist)))
    return dict(seed=seed, depth_m=cube_penetration(m, d), pairs=pairs)


def score(rows):
    # Initial validity is a hard filter, not a reward that can be offset by holding.
    return (sum(r["success30"] for r in rows), min(r["hold_sec"] for r in rows),
            float(np.mean([r["hold_sec"] for r in rows])))


def main():
    OUT.mkdir(exist_ok=False)
    m = mujoco.MjModel.from_xml_path(str(BASE))
    with np.load(SOURCE) as z:
        source = dict(z)
    save_json(OUT / "protocol.json", dict(source=str(SOURCE), source_sha256=digest(SOURCE),
        base_model_sha256=digest(BASE), development_seeds=DEV, test_seeds=TEST,
        position_grid_mm=dict(x=[0, 4, 8, 12, 16], y=[-8, -4, 0, 4, 8], z=[0, 4, 8, 12]),
        shortlist_count=5, short_seconds=8, full_seconds=40,
        only_change="object initial xyz; support follows object; hand and control trajectory unchanged",
        selection="reject reset depth >1mm on development seeds; rank by min then mean hold; top5 full40s; rank success count, min hold, mean hold",
        criterion_source="scripts/probe_s08_wrap.py:rollout, unchanged", equality_constraints=int(m.neq),
        scope="Geometry reference playback, not video-derived or BC/PPO training"))
    save_json(OUT / "baseline_reset_contacts.json", [reset_contacts(m, source, s) for s in DEV])
    candidates, packets = [], {}
    for cid, offset in enumerate(itertools.product([0, 4, 8, 12, 16], [-8, -4, 0, 4, 8], [0, 4, 8, 12])):
        p = {k: v.copy() for k, v in source.items()}
        p["object_position"] += np.asarray(offset)*.001
        resets = [reset_contacts(m, p, s) for s in DEV]
        row = dict(candidate=cid, offset_mm=offset, reset_contacts=resets, tests=[])
        row["accepted_reset"] = all(r["depth_m"] <= .001 for r in resets)
        if row["accepted_reset"]:
            row["tests"] = [rollout(m, p, s, 8)[0] for s in DEV]
            packets[cid] = p
        candidates.append(row)
        print("SCAN", cid, offset, row["accepted_reset"], [r["hold_sec"] for r in row["tests"]], flush=True)
    save_json(OUT / "scan.json", candidates)
    eligible = [r for r in candidates if r["accepted_reset"]]
    if not eligible:
        save_json(OUT / "report.json", dict(status="no reset-valid candidates"))
        return
    shortlist = sorted(eligible, key=lambda r: score(r["tests"]), reverse=True)[:5]
    development = []
    for r in shortlist:
        rows = [rollout(m, packets[r["candidate"]], s, 40)[0] for s in DEV]
        development.append(dict(candidate=r["candidate"], offset_mm=r["offset_mm"], tests=rows))
        print("FULL_DEV", r["candidate"], score(rows), flush=True)
    chosen = max(development, key=lambda r: score(r["tests"]))
    p = packets[chosen["candidate"]]
    for key in source:
        if key != "object_position":
            np.testing.assert_array_equal(source[key], p[key])
    np.savez_compressed(OUT / "reference.npz", **p)
    tree = ET.parse(BASE)
    body = tree.find(".//body[@name='cube']")
    body.set("pos", " ".join(map(str, p["object_position"])))
    body.set("quat", " ".join(map(str, p["object_quaternion"])))
    support = p["object_position"]-[0, 0, cylinder_half_height(p["object_quaternion"])+.004]
    tree.find(".//body[@name='cube_support']").set("pos", " ".join(map(str, support)))
    tree.write(OUT / "model.xml", encoding="unicode")
    save_json(OUT / "selection.json", dict(chosen=chosen, full_development=development,
        reference_sha256=digest(OUT / "reference.npz"), controls_identical=True,
        frozen_before_test=True, object_position_m=p["object_position"].tolist()))
    original_step = mujoco.mj_step
    reports, flat = {}, []
    for label, packet in [("baseline", source), ("position_only", p)]:
        rows = []
        folder = OUT / label
        folder.mkdir()
        for seed in TEST:
            warnings = np.zeros(len(mujoco.MjData(m).warning), dtype=int)
            def audited_step(model, data):
                original_step(model, data)
                warnings[:] = np.maximum(warnings, data.warning.number)
                assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
            with patch("probe_s08_wrap.mujoco.mj_step", side_effect=audited_step):
                report, trace = rollout(m, packet, seed, 40)
            assert not warnings.any(), warnings
            # Recompute hold from saved trace, not only the episode report.
            assert float(trace["hold"].max()) == report["hold_sec"]
            rows.append(report)
            np.savez_compressed(folder / f"trace_{seed}.npz", **trace)
            flat.append(dict(method=label, **{k:v for k,v in report.items() if k != "final_loads"},
                             **{f"load_{i}_n":v for i,v in enumerate(report["final_loads"])}, warnings=0))
            print("TEST", label, seed, report["success30"], report["hold_sec"], report["reset_penetration_m"], flush=True)
        reports[label] = dict(successes=sum(r["success30"] for r in rows), count=len(rows), tests=rows)
    negative, trace = rollout(m, p, TEST[0], 8, disable_contacts=True)
    assert not negative["success30"] and negative["duration_sec"] < 8
    np.savez_compressed(OUT / "negative_trace.npz", **trace)
    reports["negative_control"] = negative
    reports["warning_free"] = True
    reports["source_unchanged"] = digest(SOURCE) == json.loads((OUT / "protocol.json").read_text())["source_sha256"]
    save_json(OUT / "report.json", reports)
    with (OUT / "episodes.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print("DONE", {k:(v["successes"],v["count"]) for k,v in reports.items() if isinstance(v,dict) and "count" in v}, flush=True)


if __name__ == "__main__":
    main()
