"""Development-only bounded physical correction of S09 video-derived controls."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from retarget_s09_reference import digest
from s09_suspension import rollout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--bound", type=float, default=.24)
    p.add_argument("--seconds", type=float, default=8.)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    xml = (a.input/"model.xml").read_text()
    (a.output/"model.xml").write_text(xml)
    from s09_suspension import SuspendedCylinder
    model = SuspendedCylinder(xml).model
    selected = np.r_[0, np.arange(7, 17)]
    reports = []
    for folder in sorted(a.input.glob("pose_*")):
        with np.load(folder/"reference.npz") as z:
            packet = dict(z)
        output = a.output/folder.name
        output.mkdir()
        original = packet["ctrl"].copy()
        rows = []
        best = [float("inf"), None]

        def controls(delta):
            ctrl = original.copy()
            ctrl[:, selected] += np.minimum(packet["time"]/2., 1)[:, None]*delta
            return np.clip(ctrl, *model.actuator_ctrlrange.T)

        def objective(delta):
            report, trace = rollout(dict(packet, ctrl=controls(delta)), seed=9021,
                                     seconds=a.seconds, xml=xml)
            # Physical support counts only after release; proximity alone cannot win.
            clean = ~trace["forbidden"] & (trace["penetration_m"] <= .003) & trace["released"]
            coverage = np.minimum(trace["loads"][:, 0]/.02, 1)*np.maximum.reduce(
                np.minimum(trace["loads"][:, 1:]/.02, 1), axis=1)
            score = report["best_hold_sec"] + .1*np.sum(coverage*clean)*.05
            cost = -score + 1000*max(report["max_released_penetration_m"]-.003, 0) + .01*np.linalg.norm(delta)
            row = dict(evaluation=len(rows), cost=float(cost), delta=np.asarray(delta).tolist(), **report)
            rows.append(row)
            if cost < best[0]:
                best[:] = [cost, row]
                print("BEST", folder.name, json.dumps(row), flush=True)
            return cost

        objective(np.zeros(len(selected)))
        differential_evolution(objective, [(-a.bound, a.bound)]*len(selected), maxiter=a.iterations,
                               popsize=4, seed=9021, polish=False, tol=0., x0=np.zeros(len(selected)))
        packet.update(ctrl=controls(np.asarray(best[1]["delta"])), refinement_delta=np.asarray(best[1]["delta"]),
                      refinement_actuators=selected)
        np.savez_compressed(output/"reference.npz", **packet)
        report, trace = rollout(packet, seed=9022, seconds=40., xml=xml)
        np.savez_compressed(output/"trace.npz", **trace)
        (output/"search.json").write_text(json.dumps(rows, indent=2)+"\n")
        row = dict(pose=folder.name, development_best=best[1], validation=report,
                   input_reference_sha256=digest(folder/"reference.npz"),
                   reference_sha256=digest(output/"reference.npz"))
        (output/"report.json").write_text(json.dumps(row, indent=2)+"\n")
        reports.append(row)
        (a.output/"report.json").write_text(json.dumps(reports, indent=2)+"\n")
        print("VALIDATION", folder.name, json.dumps(report), flush=True)
    (a.output/"manifest.json").write_text(json.dumps(dict(status="development_only", input=str(a.input),
        iterations=a.iterations, bound=a.bound, seconds=a.seconds, validation_jitter=0.,
        warning="Longer replay of development poses, not independent orientation generalization or formal test"), indent=2)+"\n")


if __name__ == "__main__":
    main()
