"""Test contact-aware pregrasp fitting separately from learned cap policies."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from s11_cap import CapPilot, OUT, SOURCE, REFERENCE


def fit(sim):
    sim.reset(240000)
    model, data = sim.model, sim.data
    initial = np.asarray(sim.meta["initial_ctrl"])
    pads = [model.geom(name).id for name in
            ("orca_thumb_tip_proxy_fit_02", "orca_index_tip_proxy_fit_03")]
    caps = np.flatnonzero(sim.cap_mask)
    targets = []
    distances = []
    for pad in pads:
        candidates = []
        for cap in caps:
            segment = np.zeros(6)
            distance = mujoco.mj_geomDistance(model, data, pad, int(cap), 1., segment)
            candidates.append((distance, segment.copy()))
        distance, segment = min(candidates, key=lambda item: item[0])
        direction = segment[3:] - segment[:3]
        direction /= np.linalg.norm(direction)
        targets.append(data.geom_xpos[pad] + direction * (distance + .0004))
        distances.append(distance)
    targets = np.asarray(targets)

    def residual(q):
        data.qpos[sim.qids] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        depths = np.zeros(model.ngeom)
        for contact in data.contact[:data.ncon]:
            a, b = contact.geom
            if model.geom_contype[a] == 1 and model.geom_contype[b] in (2, 4):
                depths[a] = max(depths[a], -contact.dist)
            if model.geom_contype[b] == 1 and model.geom_contype[a] in (2, 4):
                depths[b] = max(depths[b], -contact.dist)
        return np.r_[(data.geom_xpos[pads] - targets).ravel(),
                     10 * np.maximum(depths - .0005, 0), .0005 * (q - initial)]

    low, high = model.actuator_ctrlrange.T
    result = least_squares(residual, np.clip(initial, low + 1e-8, high - 1e-8),
                           bounds=(low, high), max_nfev=300,
                           ftol=1e-9, xtol=1e-9, gtol=1e-9)
    residual(result.x)
    return result.x, dict(converged=bool(result.success), nfev=result.nfev,
                          initial_pad_gaps_m=distances,
                          fitted_pad_errors_m=np.linalg.norm(data.geom_xpos[pads]-targets, axis=1).tolist(),
                          max_joint_change_rad=float(abs(result.x-initial).max()),
                          target_pad_centers=targets.tolist(), ctrl=result.x.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--count", type=int, default=2)
    args = parser.parse_args()
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for source in (Path(__file__), Path(__file__).with_name("s11_cap.py"),
                   SOURCE / "model.xml", SOURCE / "report.json", REFERENCE):
        hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        if source.suffix == ".py":
            (out / source.name).write_bytes(source.read_bytes())
    ctrl, fitted = fit(CapPilot())
    (out / "fit.json").write_text(json.dumps(fitted, indent=2)+"\n")
    print(json.dumps(fitted), flush=True)
    episodes = []
    for moving in (False, True):
        sim = CapPilot(moving=moving)
        sim.meta["initial_ctrl"] = ctrl.tolist()
        sim.ref_ctrl[:] = ctrl
        group = out / ("moving" if moving else "fixed")
        group.mkdir()
        (group / "model.xml").write_text(sim.xml)
        for seed in range(240000, 240000+args.count):
            sim.reset(seed)
            rows = []
            for _ in range(round(sim.duration / sim.dt)):
                row = sim.step()
                rows.append(row)
                if row["floor_contact"] or row["peak_force_n"] > 100:
                    break
            arrays = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
            np.savez_compressed(group / f"seed{seed}.npz", **arrays)
            record = dict(moving=moving, seed=seed, success=rows[-1]["success"],
                          released=rows[-1]["released"],
                          max_degrees=float(np.rad2deg(arrays["angle"].max())),
                          final_degrees=float(np.rad2deg(rows[-1]["angle"])),
                          final_lift_m=rows[-1]["lift"], final_hold_sec=rows[-1]["hold_sec"],
                          best_hold_sec=sim.best_hold*sim.dt, initial_depth_m=sim.initial_depth,
                          max_depth_m=sim.max_depth, peak_force_n=sim.max_force,
                          paired_steps=int(arrays["paired"].sum()), ever_bad=bool(sim.ever_bad),
                          duration_sec=rows[-1]["time"])
            episodes.append(record)
            (out / "episodes.json").write_text(json.dumps(episodes, indent=2)+"\n")
            print(json.dumps(record), flush=True)
    (out / "summary.json").write_text(json.dumps(dict(
        status="completed", kind="contact-fit mechanics probe, not PPO", config=vars(args),
        sources=hashes, fit=fitted, episodes=episodes), indent=2)+"\n")


if __name__ == "__main__":
    main()
