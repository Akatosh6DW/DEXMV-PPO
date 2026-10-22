"""Freeze development-only selection, then run paired held-out S05 evaluations."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(run):
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "completed"
    rows = [row for row in summary["rows"] if row["step"] > 0]
    best = max(rows, key=lambda row: (row["success30"], row["pad_hold_sec"], row["step"]))
    checkpoint = run / f"step_{best['step']:09d}" / "params.pkl"
    return dict(run=str(run.resolve()), step=best["step"], development_metrics=best,
                checkpoint=str(checkpoint.resolve()), sha256=digest(checkpoint))


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    width = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [center - width, center + width]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--guided", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    chosen = {"residual": select(args.residual), "guided": select(args.guided)}
    metas = [json.loads((run / "run_config.json").read_text()) for run in (args.residual, args.guided)]
    for field in ("model_sha256", "reference_sha256", "height_weight", "pad_shaping"):
        assert metas[0][field] == metas[1][field], "Unmatched environments: " + field
    bc_path = args.guided / "step_000000000" / "params.pkl"
    chosen["bc_only"] = dict(checkpoint=str(bc_path.resolve()), sha256=digest(bc_path), step=0,
                             note="T10 step zero follows BC pretraining but no online PPO updates")
    args.output.mkdir(parents=True, exist_ok=False)
    selection = dict(criteria="max development success30, then pad hold, then latest positive step",
                     chosen=chosen, eval_seeds=list(range(52000, 52032)),
                     model_sha256=metas[0]["model_sha256"], reference_sha256=metas[0]["reference_sha256"])
    # This is committed to disk before any held-out rollout is observed.
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    original_path = ROOT / "artifacts/orca_collision_refit/O20_hic53_elliptic/cube.xml"
    original = mujoco.MjModel.from_xml_path(str(original_path))
    current = mujoco.MjModel.from_xml_path(metas[0]["model"])
    unchanged_fields = ("body_pos", "body_quat", "body_mass", "body_inertia", "geom_pos", "geom_quat",
                        "geom_size", "geom_type", "geom_friction", "geom_contype", "geom_conaffinity",
                        "jnt_range", "actuator_ctrlrange", "actuator_gainprm", "eq_type", "eq_data")
    geometry = {field: bool(np.array_equal(getattr(original, field), getattr(current, field)))
                for field in unchanged_fields}
    (args.output / "geometry_invariants.json").write_text(json.dumps(dict(
        original_model_sha256=digest(original_path), selected_model_sha256=metas[0]["model_sha256"],
        unchanged=geometry, impratio=[original.opt.impratio, current.opt.impratio],
        timestep=[original.opt.timestep, current.opt.timestep],
        note="Reference/reset can change joint and object initial poses; these checks concern compiled geometry/physics"), indent=2) + "\n")
    assert all(geometry.values()), "Unexpected geometry/physical-property change"
    entries = (("reference", "reference", None), ("residual", "residual", chosen["residual"]),
               ("bc_only", "guided", chosen["bc_only"]), ("guided", "guided", chosen["guided"]))
    stats, all_episodes, initial = {}, [], None
    for label, method, choice in entries:
        output = args.output / label
        command = [sys.executable, str(ROOT / "scripts/run_s05_job.py"),
                   "--input", metas[0]["input"], "--output", str(output), "--method", method,
                   "--steps", "0", "--eval-count", "32", "--eval-start", "52000",
                   "--height-weight", str(metas[0]["height_weight"]),
                   "--pad-shaping", str(metas[0]["pad_shaping"]), "--save-all-trajectories"]
        if choice:
            assert digest(Path(choice["checkpoint"])) == choice["sha256"]
            command += ["--checkpoint", choice["checkpoint"]]
        print("RUN", label, command, flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
        directory = output / "step_000000000"
        episodes = json.loads((directory / "episodes.json").read_text())
        with np.load(directory / "initial_states.npz") as archive:
            qpos = archive["qpos"].copy()
        if initial is None:
            initial = qpos
        else:
            np.testing.assert_array_equal(qpos, initial)
        successes = sum(e["success30"] for e in episodes)
        stats[label] = dict(total=len(episodes), successes=successes, rate=successes / len(episodes),
                            wilson95=wilson(successes, len(episodes)),
                            mean_pad_hold_sec=float(np.mean([e["pad_hold_sec"] for e in episodes])),
                            mean_pair_fraction=float(np.mean([e["pad_pair_fraction"] for e in episodes])),
                            max_penetration_m=max(e["penetration_m"] for e in episodes),
                            forbidden_episodes=sum(e["forbidden"] for e in episodes),
                            peak_force_n=max(e["peak_force_n"] for e in episodes))
        all_episodes.extend(dict(method=label, **e) for e in episodes)
    with (args.output / "all_episodes.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_episodes[0]))
        writer.writeheader()
        writer.writerows(all_episodes)
    result = dict(status="completed", initial_states_identical=True, methods=stats,
                  limitation="One training seed; fixed-task held-out initial states, not unseen objects or real hardware")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    labels = list(stats)
    rates = np.array([stats[label]["rate"] for label in labels])
    intervals = np.array([stats[label]["wilson95"] for label in labels]).T
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(labels, rates * 100, color=["#64748b", "#0f766e", "#c2410c", "#7c3aed"],
                yerr=np.maximum(0, np.stack([rates - intervals[0], intervals[1] - rates])) * 100,
                capsize=4)
    axes[0].set_ylabel("Strict success (%) with Wilson 95% CI")
    axes[0].set_ylim(0, 110)
    axes[0].tick_params(axis="x", rotation=20)
    for label, run in (("Residual PPO", args.residual), ("BC + PPO", args.guided)):
        history = json.loads((run / "summary.json").read_text())["rows"]
        axes[1].plot([r["step"] for r in history], [r["success30"] * 100 for r in history],
                     marker="o", label=label)
    axes[1].set_xlabel("Online environment interactions")
    axes[1].set_ylabel("Development success (%)")
    axes[1].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    figure.tight_layout()
    figure.savefig(args.output / "success_and_training.png", dpi=160)
    plt.close(figure)
    print("PAIRED RESULTS", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
