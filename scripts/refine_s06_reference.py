"""Bounded simulator correction of the three-finger source mapping, not RL."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from probe_s06_reference import rollout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--bound", type=float, default=.24)
    p.add_argument("--seconds", type=float, default=8.)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    m = mujoco.MjModel.from_xml_path(str(a.input / "model.xml"))
    with np.load(a.input / "reference.npz") as z:
        packet = dict(z)
    original = packet["ctrl"].copy()
    selected = np.arange(7, 17)
    rows, best = [], [float("inf"), None]

    def controls(delta):
        ctrl = original.copy()
        ctrl[:, selected] += np.minimum(packet["time"]/1., 1)[:, None]*delta
        return np.clip(ctrl, *m.actuator_ctrlrange.T)

    def objective(delta):
        trial = dict(packet, ctrl=controls(delta))
        report, trace = rollout(m, trial, 6101, a.seconds)
        released = np.arange(len(trace["hold"])) >= 59
        touched = (trace["good"] > .02).sum(1)
        dirty = (trace["bad"] > .02).any(1) | trace["forbidden"] | (trace["penetration"] > .003)
        soft = np.sum(np.where(released & ~dirty, touched/3., 0))*.05
        cost = -report["hold_sec"] - .25*soft + 2000*max(report["max_penetration_m"]-.003, 0) + .01*np.linalg.norm(delta)
        row = dict(evaluation=len(rows), cost=float(cost), delta=delta.tolist(), **report)
        rows.append(row)
        if cost < best[0]:
            best[:] = [cost, row]
            print("BEST", json.dumps(row), flush=True)
            (a.output / "best_so_far.json").write_text(json.dumps(row, indent=2)+"\n")
        return cost

    objective(np.zeros(len(selected)))
    differential_evolution(objective, [(-a.bound, a.bound)]*len(selected), maxiter=a.iterations,
                           popsize=5, seed=6101, polish=False, tol=0., x0=np.zeros(len(selected)))
    delta = np.array(best[1]["delta"])
    packet.update(ctrl=controls(delta), refinement_delta=delta, refinement_actuators=selected)
    np.savez_compressed(a.output / "reference.npz", **packet)
    (a.output / "model.xml").write_bytes((a.input / "model.xml").read_bytes())
    reports = []
    for seed in range(6110, 6114):
        report, trace = rollout(m, packet, seed, 40.)
        reports.append(report)
        np.savez_compressed(a.output / f"trace_{seed}.npz", **trace)
        print("VALIDATION", json.dumps(report), flush=True)
    (a.output / "search.json").write_text(json.dumps(rows, indent=2)+"\n")
    (a.output / "report.json").write_text(json.dumps(dict(input=str(a.input), bound=a.bound,
        seconds=a.seconds, iterations=a.iterations, best=best[1], validation=reports,
        warning="Development reference correction, not PPO or held-out performance"), indent=2)+"\n")


if __name__ == "__main__":
    main()
