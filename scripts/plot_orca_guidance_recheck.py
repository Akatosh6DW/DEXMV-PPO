"""Plot all training seeds, with initial competence separate from PPO gain."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/orca_guidance_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True, type=Path)
    args = p.parse_args()
    result = json.loads((args.suite / "analysis.json").read_text())
    colors = dict(pure="#176a9a", bc="#b65b10", residual="#247c4e")
    names = dict(pure="PPO from scratch", bc="BC init + PPO", residual="Video + residual PPO")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for i, reward in enumerate(("shaped", "hold_only")):
        for j, arm in enumerate(("pure", "bc", "residual")):
            ax = axes[i, j]
            runs = [r for r in result["runs"] if r["arm"] == arm and r["reward"] == reward]
            arrays = []
            for run in runs:
                x = np.asarray([r["step"] for r in run["curve"]])
                y = np.asarray([r["success30"] for r in run["curve"]])
                ax.plot(x, y, alpha=.35, lw=1, color=colors[arm])
                arrays.append(y)
            ax.plot(x, np.mean(arrays, axis=0), lw=2.5, color=colors[arm], label="Mean; thin lines = seeds")
            ax.axhline(.75, color="#888888", ls=":", lw=.8)
            ax.set_title(f"{reward}: {names[arm]}")
            ax.set_ylim(-.02, 1.02)
            ax.grid(alpha=.2)
            if i == 1:
                ax.set_xlabel("PPO environment steps (BC/preparation extra)")
            if j == 0:
                ax.set_ylabel("30 s clean-grasp success")
            ax.legend(fontsize=8)
    fig.suptitle("All confirmatory seeds; step 0 is not PPO learning", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.suite / "learning_curves.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, reward in zip(axes, ("shaped", "hold_only")):
        for arm in ("pure", "bc", "residual"):
            runs = [r for r in result["runs"] if r["arm"] == arm and r["reward"] == reward]
            values = np.asarray([[r["success30"] for r in run["curve"]] for run in runs])
            values -= values[:, :1]
            ax.plot(x, values.mean(axis=0), label=names[arm], color=colors[arm])
        ax.axhline(0, color="#888888", lw=.8)
        ax.set_title(reward)
        ax.set_xlabel("PPO environment steps")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Success minus each run's initial success")
    fig.suptitle("PPO improvement (interpret alongside absolute success)")
    fig.tight_layout()
    fig.savefig(args.suite / "learning_gain.png", dpi=150)


if __name__ == "__main__":
    main()
