"""Bounded simulator refinement of a video-derived S05 reference (not PPO)."""

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from diagnose_orca_grasp import model_indices
from probe_hic53_orca import native_rollout
from s05_pad_contacts import pad_contacts


def evaluate(model, packet, ctrl, seed, seconds):
    trace, report = native_rollout(model, packet["object_position"], packet["object_quaternion"],
                                   ctrl, seed, seconds=seconds, initial_ctrl=packet["initial_ctrl"])
    data = mujoco.MjData(model)
    hold = best = pair_count = 0
    bad_count = 0
    for i in range(59, len(trace["qpos"]), 2):
        data.qpos[:], data.qvel[:], data.ctrl[:] = trace["qpos"][i], trace["qvel"][i], trace["ctrl"][i]
        data.mocap_pos[:] = trace["mocap"][i]
        mujoco.mj_forward(model, data)
        good, bad, _ = pad_contacts(model, data)
        paired = (good > .02).all() and not (bad > .02).any()
        valid = paired and trace["hold"][i] > 0 and trace["penetration"][i] <= .003
        pair_count += paired
        bad_count += (bad > .02).any()
        hold = hold + 1 if valid else 0
        best = max(best, hold)
    report.update(pad_hold_sec=best * .1, pad_pair_samples=int(pair_count), nonpad_samples=int(bad_count))
    return report, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bound", type=float, default=.12)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--probe-seconds", type=float, default=8.)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(args.input / "model.xml"))
    with np.load(args.input / "reference.npz", allow_pickle=False) as archive:
        packet = {k: archive[k].copy() for k in archive.files}
    original = packet["ctrl"].copy()
    chosen = np.array([0, 10, 11, 12, 13, 14, 15, 16])
    low, high = model.actuator_ctrlrange.T
    rows = []
    best = [float("inf"), None]

    def controls(delta):
        ctrl = original.copy()
        blend = np.minimum(packet["time"] / 1.0, 1.)
        ctrl[:, chosen] += blend[:, None] * delta
        return np.clip(ctrl, low, high)

    def objective(delta):
        ctrl = controls(delta)
        report, _ = evaluate(model, packet, ctrl, 5011, args.probe_seconds)
        penalty = 1000 * max(report["max_penetration_m"] - .003, 0)
        cost = (-report["pad_hold_sec"] - .02 * report["pad_pair_samples"]
                - .05 * report["best_hold_sec"] + penalty)
        row = dict(evaluation=len(rows), cost=cost, delta=delta.tolist(), **report)
        rows.append(row)
        if cost < best[0]:
            best[:] = [cost, row]
            print("BEST", json.dumps(row), flush=True)
        return cost

    objective(np.zeros(len(chosen)))
    if not args.baseline_only:
        differential_evolution(objective, [(-args.bound, args.bound)] * len(chosen),
                               maxiter=args.iterations, popsize=5, seed=5001, polish=False, tol=0.)
    delta = np.array(best[1]["delta"])
    ctrl = controls(delta)
    packet.update(ctrl=ctrl, refinement_delta=delta, refinement_actuators=chosen)
    np.savez_compressed(args.output / "reference.npz", **packet)
    (args.output / "model.xml").write_bytes((args.input / "model.xml").read_bytes())
    reports = []
    for seed in range(5020, 5028):
        report, trace = evaluate(model, packet, ctrl, seed, 40.)
        reports.append(report)
        np.savez_compressed(args.output / f"probe_{seed}.npz", **trace)
        print("VALIDATION", json.dumps(report), flush=True)
    (args.output / "search.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "report.json").write_text(json.dumps(dict(
        input=str(args.input), bound=args.bound, iterations=args.iterations,
        probe_seconds=args.probe_seconds, best=best[1], probes=reports,
        warning="Development search, not trained PPO or independent test. Pad metrics use 100ms saved-state reconstruction."),
        indent=2) + "\n")
    with (args.output / "metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=[k for k in rows[0] if k != "delta"])
        writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k != "delta"} for row in rows)


if __name__ == "__main__":
    main()
