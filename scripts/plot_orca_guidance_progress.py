"""Explicitly partial learning curves; never pads unfinished training runs."""

import argparse
import csv
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
    p.add_argument("--suite", type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads((args.suite / "protocol.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    completed = 0
    for i, reward in enumerate(protocol["rewards"]):
        for j, arm in enumerate(protocol["arms"]):
            ax = axes[i, j]
            panel_completed = 0
            for k, seed in enumerate(protocol["seeds"]):
                std = protocol["chosen"][arm]["std"]
                directory = args.suite / f"{reward}_{arm}_std{std}_seed{seed}"
                path = directory / "metrics.csv"
                if not path.exists():
                    continue
                with path.open() as f:
                    rows = list(csv.DictReader(f))
                try:
                    x = np.array([float(r["step"]) for r in rows])
                    y = np.array([float(r["success30"]) for r in rows])
                except (TypeError, ValueError, KeyError):
                    continue  # The trainer can be replacing this CSV during a read.
                if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
                    continue
                done = (directory / "summary.json").exists() and x[-1] == protocol["steps"]
                panel_completed += int(done)
                ax.plot(x/1e6, y, color=f"C{k}", ls="-" if done else "--", lw=1.4,
                        label=f"seed {seed}" + ("" if done else " (partial)"))
                ax.scatter(x[-1]/1e6, y[-1], color=f"C{k}", s=15)
            completed += panel_completed
            ax.set_title(f"{reward} / {arm}: {panel_completed}/5 complete")
            ax.set_xlim(0, protocol["steps"]/1e6)
            ax.set_ylim(-.03, 1.03)
            ax.axvline(.655360, color="#777777", ls=":", lw=.8)
            ax.grid(alpha=.2)
            if ax.lines and ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
            if i == 1:
                ax.set_xlabel("PPO steps (millions)")
            if j == 0:
                ax.set_ylabel("30 s grasp success")
    fig.suptitle(f"IN PROGRESS: {completed}/30 runs complete; no extrapolation or partial-group means")
    fig.tight_layout()
    fig.savefig(args.suite / "progress.png", dpi=130)
    plt.close(fig)
    print(json.dumps(dict(completed=completed, total=30, plot=str(args.suite / "progress.png"))))


if __name__ == "__main__":
    main()
