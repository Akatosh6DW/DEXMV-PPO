"""Audit matched protocols and summarize all three arms without hiding failures."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def curve_summary(rows):
    x = np.array([float(r["step"]) for r in rows])
    y = np.array([float(r["success30"]) for r in rows])
    if not len(x) or np.any(np.diff(x) <= 0):
        raise ValueError("Missing or repeated evaluations")
    index = int(np.argmax(y))
    converged = np.flatnonzero((y[1:] >= .75) & (y[:-1] >= .75))
    return dict(first=float(y[0]), last=float(y[-1]), max=float(y[index]),
                best_step=int(x[index]), actual_steps=int(x[-1]),
                auc=float(np.trapezoid(y, x)/x[-1]) if x[-1] else None,
                two_eval_75_step=int(x[converged[0]+1]) if len(converged) else None)


def diagnose(trace_path, bound=.08):
    with np.load(trace_path, allow_pickle=False) as z:
        alive = z["alive"][:, -1]
        contact = (z["thumb"] > 0) & (z["contacts"] >= 2)
        clean = (z["forbidden"] == 0) & contact & z["alive"]
        above = z["cube_z"] > z["spawn_z"]-.01
        active = z["released"] > 0
        # Include the action that causes termination, exclude diagnostic tails.
        active_action = np.concatenate([np.ones((len(alive), 1), bool), z["alive"][:, :-1]], axis=1)
        active_requested = np.abs(z["requested_residual"])[active_action]
        active_effective = np.abs(z["effective_residual"])[active_action]
        return dict(dropped_fraction=float((~alive).mean()),
                    retained_but_forbidden_fraction=float((alive & (z["forbidden"][:, -1] > 0)).mean()),
                    retained_but_missing_finger_pair_fraction=float((alive & ~contact[:, -1]).mean()),
                    retained_but_too_low_fraction=float((alive & ~above[:, -1]).mean()),
                    clean_supported_free_frame_fraction=float((clean & above)[active].mean()),
                    requested_residual_max_rad=float(np.abs(z["requested_residual"]).max()),
                    effective_residual_max_rad=float(np.abs(z["effective_residual"]).max()),
                    active_action_requested_max_rad=float(active_requested.max(initial=0)),
                    active_action_effective_max_rad=float(active_effective.max(initial=0)),
                    active_action_near_bound_fraction=float((active_requested > bound*.95).mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("suite", type=Path)
    args = p.parse_args()
    protocol = json.loads((args.suite/"protocol.json").read_text())
    records = []
    for shape in protocol["shapes"]:
        for seed in protocol["seeds"]:
            paired = []
            for arm in protocol["arms"]:
                run = args.suite/f"{shape}_{arm}_seed{seed}"
                if not (run/"summary.json").exists():
                    raise ValueError(f"Incomplete run: {run}")
                meta = json.loads((run/"run_config.json").read_text())
                with (run/"metrics.csv").open() as f:
                    rows = list(csv.DictReader(f))
                record = dict(task=shape, arm=arm, seed=seed, run=str(run), **curve_summary(rows))
                record["bc_sec"] = meta["bc_sec"]
                record["wall_sec"] = float(rows[-1]["elapsed_sec"])
                record["terminal_diagnostics"] = diagnose(run/f"step_{record['actual_steps']:09d}/trajectories.npz", meta["bound"])
                if record["actual_steps"] != protocol["steps"]:
                    raise ValueError(f"Budget mismatch: {run}")
                if meta["precision"] != "highest":
                    raise ValueError("Precision was not fixed")
                paired.append((meta, (run/"env_config.json").read_bytes()))
                records.append(record)
            first, config = paired[0]
            for meta, other_config in paired[1:]:
                for key in ("seed", "steps", "evals", "eval_start", "eval_count", "model_sha256",
                            "learning_rate", "init_std", "precision", "initialization"):
                    if meta[key] != first[key]:
                        raise ValueError(f"Unmatched {key}: {shape}, seed{seed}")
                if config != other_config:
                    raise ValueError("Unmatched environment/reward config")
                if protocol.get("confirmatory") and meta["source_sha256"] != first["source_sha256"]:
                    raise ValueError("Source changed during confirmatory training")
            methods = {m["arm"]: m for m, _ in paired}
            if "bc" in methods and "residual" in methods:
                if methods["bc"]["dataset_provenance"]["reference_sha256"] != methods["residual"]["reference_sha256"]:
                    raise ValueError("BC and residual did not share the same reference source")
    aggregates = []
    for shape in protocol["shapes"]:
        for arm in protocol["arms"]:
            group = [r for r in records if r["task"] == shape and r["arm"] == arm]
            aggregates.append(dict(task=shape, arm=arm, n_training_seeds=len(group),
                                   mean_auc=float(np.mean([r["auc"] for r in group])),
                                   mean_last=float(np.mean([r["last"] for r in group])),
                                   mean_best=float(np.mean([r["max"] for r in group])),
                                   converged_count=sum(r["two_eval_75_step"] is not None for r in group)))
    result = dict(protocol=protocol, runs=records, aggregates=aggregates,
                  selection_rule="earliest maximum validation success30; terminal also evaluated",
                  warning="validation scores are not held-out performance; episodes are not independent training seeds")
    (args.suite/"analysis.json").write_text(json.dumps(result, indent=2))
    with (args.suite/"analysis.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
