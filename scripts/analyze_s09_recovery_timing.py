"""Summarize complete paired teacher searches without treating them as policies."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.run/"protocol.json").read_text())
    completed = json.loads((args.run/"summary.json").read_text())
    if completed["status"] != "completed":
        raise ValueError("The paired search must finish before analysis")
    all_rows, baselines, hashes = [], {}, {}
    for prefix in (40, 60):
        teacher = args.run/f"prefix_{prefix:03d}"/"teacher"
        rows = json.loads((teacher/"results.json").read_text())
        if [r["seed"] for r in rows] != protocol["seeds"]:
            raise ValueError("Cases or ordering changed")
        for case in rows:
            folder = teacher/f"seed{case['seed']}"
            history = json.loads((folder/"search.json").read_text())
            expected = 1 if protocol["smoke"] else 161
            if len(history) != expected or case["search_evaluations"] != expected:
                raise ValueError("Teacher search budget changed")
            best = min(history, key=lambda r: r["cost"])
            np.testing.assert_array_equal(best["delta"], case["selected_delta"])
            for label in ("unchanged_bc", "selected_offset"):
                path = folder/f"{label}.npz"
                with np.load(path, allow_pickle=False) as z:
                    trace = dict(z)
                hashes[str(path.relative_to(args.run))] = hashlib.sha256(path.read_bytes()).hexdigest()
                released = trace["released"].astype(bool)
                success = bool(trace["success30"][-1] and trace["alive"][-1])
                if success != case[label]["success30"]:
                    raise ValueError("Stored success label differs from trace")
                if label == "unchanged_bc":
                    if prefix == 40:
                        baselines[case["seed"]] = trace
                    else:
                        # Both branches must follow the same physical baseline
                        # after their common absolute 3-second boundary.
                        for key in trace:
                            np.testing.assert_array_equal(baselines[case["seed"]][key][20:], trace[key])
                all_rows.append(dict(prefix_sec=prefix*.05, seed=case["seed"], mode=case["mode"],
                    controller=label, success30=success,
                    best_hold_sec=float(trace["best_hold_sec"].max()),
                    final_valid=bool(trace["valid"][-1]), final_time_sec=float(trace["time"][-1]),
                    max_released_penetration_mm=1000*float(np.max(trace["penetration_m"][released], initial=0)),
                    max_all_time_penetration_mm=1000*float(trace["penetration_m"].max()),
                    released_forbidden=bool(np.any(trace["forbidden"][released])),
                    released_rope_force_n=float(np.max(trace["rope_force_n"][released], initial=0)),
                    peak_force_n=float(trace["peak_force_n"].max())))
    output = args.run/"report"
    output.mkdir(exist_ok=True)
    with (output/"cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    lines = ["# Recovery Teacher Timing: Training-Case Diagnostic", "",
        "Not a BC/PPO success-rate benchmark. Same historical failure cases and search budget;",
        "only intervention time changes. No policy updates, new videos, or relaxed success thresholds.", "",
        "| Time | Seed | Controller | Success30 | Best hold (s) | Released depth (mm) | Peak force (N) |",
        "|---:|---:|---|---:|---:|---:|---:|"]
    for row in all_rows:
        lines.append(f"| {row['prefix_sec']} | {row['seed']} | {row['controller']} | "
                     f"{int(row['success30'])} | {row['best_hold_sec']:.3f} | "
                     f"{row['max_released_penetration_mm']:.3f} | {row['peak_force_n']:.3f} |")
    lines.extend(["", "All baseline trace arrays match after the common 3-second boundary.",
                  "Selected offsets match the minimum search cost, not a choice based on long validation.",
                  "All cases are retained. A teacher plan that succeeds is not proof a student can reproduce it."])
    for row in completed["rows"]:
        lines.append(f"\n{row['prefix_sec']}s: unchanged {row['unchanged_successes']}/{row['cases']}, "
                     f"corrected {row['corrected_successes']}/{row['cases']}; "
                     f"{row['total_native_control_steps']} native control steps, including search and replay.")
    (output/"results.md").write_text("\n".join(lines)+"\n")
    evidence = dict(status="verified", training_only=True, policy_updates=0,
        smoke=protocol["smoke"], baseline_overlap_exact=True, selected_cost_minimum=True,
        trace_sha256=hashes, summary_sha256=hashlib.sha256((args.run/"summary.json").read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/"evidence.json").write_text(json.dumps(evidence, indent=2)+"\n")
    print(json.dumps(completed, indent=2))


if __name__ == "__main__":
    main()
