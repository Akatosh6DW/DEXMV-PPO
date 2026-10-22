"""Plot measured learning curves, with BC shown only at zero PPO interactions."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    args = p.parse_args()
    results = json.loads((args.suite / "results.json").read_text())
    if not results["complete"]:
        raise ValueError("Finish the registered pilot before plotting the comparison")
    colors = dict(bc="#777777", ppo="#218752", guided="#c56c22", residual="#286bb4")
    names = dict(bc="BC only", ppo="PPO", guided="BC+PPO", residual="Residual PPO")
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True, constrained_layout=True)
    handles, labels = [], []
    for record in results["records"]:
        with (args.suite / record["run"] / "metrics.csv").open() as stream:
            rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        method, tactile = record["method"], record["tactile"]
        label = names[method] + (" + touch" if tactile == "on" else " (no touch)")
        style = dict(color=colors[method], linestyle="-" if tactile == "on" else "--",
                     marker="o" if tactile == "on" else "s", markersize=4, linewidth=1.8)
        if method == "bc":
            style["linestyle"] = "None"
        for axis, key, scale in zip(axes, ("success30", "mean_hold", "active_force_max_n"), (100., 1., 1.)):
            line, = axis.plot([r["step"] / 1000 for r in rows], [r[key] * scale for r in rows],
                              label=label, **style)
            if axis is axes[0]:
                handles.append(line)
                labels.append(label)
    axes[0].set_ylabel("30 s + valid final frame (%)")
    axes[0].set_ylim(-4, 105)
    axes[1].set_ylabel("Mean longest valid hold (s)")
    axes[1].set_ylim(-1, 40)
    axes[2].set_ylabel("Peak finger normal force (N)")
    axes[2].set_xlabel("PPO interactions (thousands); BC points have no PPO budget")
    for axis in axes:
        axis.grid(True, alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(handles, labels, ncol=2, loc="upper left", fontsize=8)
    fig.suptitle("HIC20 / 53 mm source object: single-seed native-physics pilot\n"
                 "16 paired evaluation states; no multi-seed uncertainty estimate", fontsize=12)
    path = args.suite / "learning_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
