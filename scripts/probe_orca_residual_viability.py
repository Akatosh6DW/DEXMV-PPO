"""Bounded native-control search to test residual authority, not a PPO policy."""

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from orca_residual_controls import load_reference
from prepare_hic_orca_object import digest
from probe_hic53_orca import native_rollout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--generations", type=int, default=4)
    p.add_argument("--population", type=int, default=48)
    p.add_argument("--search-seconds", type=float, default=22.)
    args = p.parse_args()
    if args.generations < 1 or args.population < 8 or not np.isfinite(args.search_seconds) or not 3 < args.search_seconds <= 40:
        p.error("Invalid search budget")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "probe_source.py").write_bytes(Path(__file__).read_bytes())
    model = mujoco.MjModel.from_xml_path(str(args.model))
    with np.load(args.reference) as z:
        packet = {k: z[k] for k in z.files}
    base, _ = load_reference(args.reference, *model.actuator_ctrlrange.T, packet["initial_ctrl"], .05, 800, .1)
    selected = np.arange(10, 17)
    ramp = np.clip((np.arange(800) + 1) * .05 - 1., 0, .5) / .5
    def controls(delta):
        correction = np.zeros(17)
        correction[selected] = delta
        return np.clip(base + ramp[:, None] * correction, *model.actuator_ctrlrange.T)
    def evaluate(delta, seeds, seconds):
        commands = controls(delta)
        pairs = [native_rollout(model, packet["object_position"], packet["object_quaternion"],
                                commands, seed, seconds=seconds, initial_ctrl=packet["initial_ctrl"])
                 for seed in seeds]
        checks = [pair[1] for pair in pairs]
        value = min(c["best_hold_sec"] for c in checks) + .01 * np.mean([c["duration_sec"] for c in checks])
        if max(c["max_penetration_m"] for c in checks) > .005:
            value = -1.
        return float(value), pairs
    rng = np.random.default_rng(701)
    mean, sigma, best, records = np.zeros(7), np.full(7, .04), None, []
    started = time.monotonic()
    for generation in range(args.generations):
        candidates = np.clip(rng.normal(mean, sigma, size=(args.population, 7)), -.08, .08)
        candidates[0] = mean
        candidates[1] = np.zeros(7) if best is None else best["delta"]
        rows = []
        for i, delta in enumerate(candidates):
            value, pairs = evaluate(delta, [1100, 1101], args.search_seconds)
            row = dict(generation=generation, candidate=i, delta=delta.tolist(), value=value,
                       checks=[v[1] for v in pairs])
            rows.append(row)
            if best is None or value > best["value"]:
                best = row
        elite = sorted(rows, key=lambda row: row["value"], reverse=True)[:max(4, args.population // 6)]
        elite_delta = np.array([row["delta"] for row in elite])
        mean, sigma = elite_delta.mean(0), np.maximum(elite_delta.std(0), .007)
        records.extend(rows)
        (args.output / "search.json").write_text(json.dumps(records, indent=2) + "\n")
        print("GEN", generation, json.dumps(best), flush=True)
    checks = []
    for seed in range(1200, 1208):
        _, pairs = evaluate(best["delta"], [seed], 40.)
        trace, check = pairs[0]
        checks.append(check)
        np.savez_compressed(args.output / f"native_seed{seed}.npz", **trace)
    np.savez_compressed(args.output / "diagnostic_commands.npz", ctrl=controls(best["delta"]),
                        base=base, selected=selected, delta=best["delta"], ramp=ramp)
    report = dict(kind="native bounded constant residual search, NOT PPO and NOT source demonstration",
                  model_sha256=digest(args.model), reference_sha256=digest(args.reference),
                  residual_bound_rad=.08, selected_actuators=selected.tolist(), ramp_sec=[1., 1.5],
                  best=best, validation_checks=checks, elapsed_sec=time.monotonic() - started,
                  search_seconds=args.search_seconds,
                  not_for_bc_dataset=True, no_source_reference_modified=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("VALIDATION", json.dumps(checks), flush=True)


if __name__ == "__main__":
    main()
