"""Inspect a stopped contact-quality gate without changing scores or physics."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from prepare_hic_orca_object import digest


def inspect(run):
    cfg = json.loads((run / "run_config.json").read_text())
    result_path = run / "formal_result.json"
    result = json.loads(result_path.read_text())
    trace_path = run / f"step_{int(result['final_steps']):09d}/trajectories.npz"
    geometry_path = trace_path.parent / "geometry_audit.json"
    geometry = json.loads(geometry_path.read_text())
    with np.load(trace_path) as packet:
        t = dict(packet)
    if not all(np.isfinite(v).all() for v in t.values()):
        raise ValueError("Nonfinite trace cannot be reviewed as ordinary policy failure")
    if digest(trace_path) != geometry["trace_sha256"]:
        raise ValueError("Geometry audit is stale")
    active, alive = t["active"].astype(bool), t["alive"].astype(bool)
    released = active & (t["released"] > 0)
    dirty = released & (t["forbidden"] > 0)
    np.testing.assert_allclose(dirty.sum() / released.sum(), result["forbidden_fraction"])
    valid = (alive & released & (t["thumb"] > 0) & (t["contacts"] >= 2) & (t["forbidden"] == 0)
             & (t["cube_z"] > t["spawn_z"] - .01))
    count, longest = np.zeros(len(active), int), np.zeros(len(active), int)
    for step in range(800):
        count = np.where(valid[:, step], count + 1, 0)
        longest = np.maximum(longest, count)
        np.testing.assert_allclose(t["hold"][:, step], count * .05, atol=4e-6)
    success = (longest >= 600) & valid[:, -1]
    np.testing.assert_allclose(success.mean(), result["last"])
    model = mujoco.MjModel.from_xml_path(cfg["model"])
    data, cube = mujoco.MjData(model), model.geom("cube").id
    episodes = []
    for episode in np.flatnonzero(dirty.any(axis=1)):
        steps = np.flatnonzero(dirty[episode])
        last = int(steps[-1])
        for key in ("qpos", "qvel", "ctrl"):
            getattr(data, key)[:] = t[key][episode, last]
        data.mocap_pos[:] = t["mocap"][episode, last]
        mujoco.mj_forward(model, data)
        contacts = [dict(geom=model.geom(c.geom2 if c.geom1 == cube else c.geom1).name,
                         penetration_mm=max(-float(c.dist), 0.) * 1000)
                    for c in data.contact if cube in (c.geom1, c.geom2)]
        episodes.append(dict(episode=int(episode), first_forbidden_s=float((steps[0] + 1) * .05),
                             last_forbidden_s=float((steps[-1] + 1) * .05), forbidden_steps=len(steps),
                             max_valid_hold_s=float(longest[episode] * .05), success=bool(success[episode]),
                             final_forbidden=bool(t["forbidden"][episode, -1]), last_snapshot_contacts=contacts))
    return dict(run=str(run.resolve()), result_sha256=digest(result_path), trace_sha256=digest(trace_path),
                geometry_sha256=digest(geometry_path), source_sha256=digest(Path(__file__)),
                released_forbidden_fraction=float(result["forbidden_fraction"]),
                max_penetration_mm=geometry["max_penetration_m"] * 1000,
                successful_episodes=int(success.sum()), total_episodes=len(success), dirty_episodes=episodes,
                all_hold_counters_recomputed=True, final_dirty_success_count=int(np.sum(success & (t["forbidden"][:, -1] > 0))),
                geometry_scope="Saved float32 pose collision distances; no force equivalence claim",
                decision_required=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing inspection evidence")
    result = inspect(args.run)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
