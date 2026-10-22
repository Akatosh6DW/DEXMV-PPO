"""Plot the actual base/correction composition and all matched learning curves."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"pure": "#2377a4", "bc": "#c78b25", "residual": "#26845d"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", type=Path)
    p.add_argument("--trace", type=Path)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--bound", type=float, default=.08)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.suite:
        analysis = json.loads((args.suite/"analysis.json").read_text())
        shapes = analysis["protocol"]["shapes"]
        fig, axes = plt.subplots(1, len(shapes), figsize=(5*len(shapes), 4), squeeze=False)
        for ax, shape in zip(axes.ravel(), shapes):
            for arm in analysis["protocol"]["arms"]:
                curves = []
                for run in analysis["runs"]:
                    if run["task"] != shape or run["arm"] != arm:
                        continue
                    with (Path(run["run"])/"metrics.csv").open() as f:
                        rows = list(csv.DictReader(f))
                    x = np.array([float(r["step"]) for r in rows])
                    y = np.array([float(r["success30"]) for r in rows])
                    ax.plot(x, y, color=COLORS[arm], alpha=.2, linewidth=1)
                    curves.append(y)
                ax.plot(x, np.mean(curves, axis=0), color=COLORS[arm], label=arm, linewidth=2)
            ax.set(title=shape, xlabel="PPO environment steps", ylabel="Validation success (30s + retained)", ylim=(-.03, 1.03))
            ax.grid(alpha=.2)
            ax.legend()
        fig.tight_layout()
        fig.savefig(args.output/"learning_curves.png", dpi=160)
        plt.close(fig)
    if args.trace:
        with np.load(args.trace, allow_pickle=False) as z:
            base, ctrl, delta = (z[k][args.episode] for k in ("video_base", "ctrl", "effective_residual"))
            desired = z["desired_ctrl"][args.episode]
            requested = z["requested_residual"][args.episode]
            alive = z["alive"][args.episode]
        np.testing.assert_allclose(ctrl-base, delta, atol=1e-6)
        active_action = np.concatenate([np.ones(1, dtype=bool), alive[:-1]])
        active_peak = np.max(np.abs(delta[active_action]), axis=0)
        joint = int(np.argmax(active_peak))
        t = (np.arange(len(base))+1)*.05
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(t, base[:, joint], label="video base", color="#555555")
        axes[0].plot(t, desired[:, joint], label="base + requested correction (limited)", color="#c78b25")
        axes[0].plot(t, ctrl[:, joint], label="actuator target after slew limit", color="#26845d", linestyle="--")
        axes[0].fill_between(t, base[:, joint]-args.bound, base[:, joint]+args.bound, alpha=.12, color="#26845d")
        axes[0].set(title=f"Actuator {joint}: target angles, not measured joint angles", ylabel="rad")
        axes[1].plot(t, requested[:, joint], label="requested residual", color="#c78b25")
        axes[1].plot(t, delta[:, joint], label="effective residual", color="#26845d", linestyle="--")
        axes[1].axhline(args.bound, color="#b44", linestyle=":")
        axes[1].axhline(-args.bound, color="#b44", linestyle=":")
        axes[1].set(xlabel="time (s)", ylabel="rad", ylim=(-args.bound*1.2, args.bound*1.2))
        for ax in axes:
            failed = np.flatnonzero(~alive)
            if len(failed):
                ax.axvspan(t[failed[0]], t[-1], color="#b44", alpha=.06,
                           label="after terminal drop: diagnostic only")
            ax.grid(alpha=.2)
            ax.legend()
        fig.tight_layout()
        fig.savefig(args.output/"residual_composition.png", dpi=160)
        plt.close(fig)
        (args.output/"composition.json").write_text(json.dumps(dict(trace=str(args.trace),
                    episode=args.episode, actuator_index=joint, bound_rad=args.bound,
                    actuator_selection="largest active-action effective residual; excludes post-terminal diagnostic tail",
                    active_peak_rad=float(active_peak[joint])), indent=2))


if __name__ == "__main__":
    main()
