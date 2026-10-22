"""Plot every orientation group from an already completed frozen confirmation."""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/s09-mode-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    with (args.report/"orientation_modes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    groups = ["bc", "bc_ppo", "bc_ppo_last", "reference", "residual"]
    labels = ["BC", "BC + PPO selected", "BC + PPO last", "Reference", "Residual PPO"]
    rates = np.empty((len(groups), 6))
    lookup = {(r["group"], int(r["mode"])): r for r in rows}
    if len(lookup) != len(rows) or len(rows) != 30:
        raise ValueError("Expected exactly five groups and six modes, without duplicates")
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    for i, group in enumerate(groups):
        for mode in range(6):
            row = lookup[(group, mode)]
            total, successes = int(row["policy_episode_pairs"]), int(row["successes"])
            if not 0 <= successes <= total or total == 0:
                raise ValueError("Invalid orientation counts")
            rates[i, mode] = successes/total*100
            if not np.isclose(rates[i, mode]/100, float(row["success_rate"])):
                raise ValueError("Stored rate disagrees with counts")
            ax.text(mode, i, f"{rates[i, mode]:.1f}%\n{successes}/{total}",
                    ha="center", va="center", color="black", fontsize=10)
    im = ax.imshow(rates, vmin=0, vmax=100, cmap="YlGn", aspect="auto", alpha=.65)
    ax.set(xticks=range(6), xticklabels=[f"Mode {i}" for i in range(6)],
           yticks=range(len(groups)), yticklabels=labels,
           title="Strict success by initial-orientation reference class",
           xlabel="Classes contain continuous orientations, not six fixed poses")
    fig.colorbar(im, ax=ax, label="Success (%)")
    path = args.report/"orientation_transfer.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
