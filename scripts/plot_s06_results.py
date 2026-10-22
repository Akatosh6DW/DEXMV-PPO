"""Plot held-out performance separately from development training and IK error."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--similarity", type=Path, required=True)
    a = p.parse_args()
    summary = json.loads((a.evaluation / "summary.json").read_text())
    selection = json.loads((a.evaluation / "selection.json").read_text())
    similarity = json.loads((a.similarity / "report.json").read_text())
    names, labels = ["reference", "residual", "bc_only", "guided"], ["Reference", "Residual PPO", "BC only", "BC + PPO"]
    colors = ["#787878", "#237f63", "#bd8036", "#3b72ad"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    values = [summary["methods"][n]["success_rate"]*100 for n in names]
    lower = [v-summary["methods"][n]["wilson95"][0]*100 for n, v in zip(names, values)]
    upper = [summary["methods"][n]["wilson95"][1]*100-v for n, v in zip(names, values)]
    axs[0, 0].bar(labels, values, color=colors, yerr=[lower, upper], capsize=4)
    axs[0, 0].set(ylim=(0, 110), ylabel="Success (%)", title="Independent initial states; Wilson 95% CI")
    for x, n in enumerate(names):
        row = summary["methods"][n]
        axs[0, 0].text(x, 103, f"{row['successes']}/{row['count']}", ha="center")
    for name, color in (("residual", colors[1]), ("guided", colors[3])):
        run = Path(selection["selected"][name]["checkpoint"]).parent.parent
        with (run / "metrics.csv").open() as f:
            rows = list(csv.DictReader(f))
        axs[0, 1].plot([float(r["step"]) for r in rows], [float(r["success30"])*100 for r in rows],
                       marker="o", color=color, label=name)
        selected = selection["selected"][name]["development"]
        axs[0, 1].scatter([selected["step"]], [selected["success30"]*100], marker="D", s=85,
                          facecolors="none", edgecolors=color, linewidths=2)
    axs[0, 1].set(xlabel="Online interaction steps", ylabel="Success (%)", ylim=(-5, 105),
                  title="Development: four states; diamonds = selected")
    axs[0, 1].legend()
    x = np.arange(3)
    for offset, key, label, color in ((-.25, "old_reference_errors_mm", "Old reference", colors[0]),
                                      (0, "new_reference_errors_mm", "Corrected reference", colors[2]),
                                      (.25, "valid_state_mean_errors_mm", "Trained valid states", colors[1])):
        axs[1, 0].bar(x+offset, similarity[key], width=.25, label=label, color=color)
    axs[1, 0].set(xticks=x, xticklabels=["Thumb", "Index", "Middle"], ylabel="Position error (mm)",
                  title="Source-relative targets; not full human pose error")
    axs[1, 0].legend(fontsize=8)
    for i, (name, color) in enumerate(zip(names, colors)):
        rows = json.loads((a.evaluation / name / "step_000000000/episodes.json").read_text())
        offsets = np.linspace(-.14, .14, len(rows))
        axs[1, 1].scatter(i+offsets, [r["pad_hold_sec"] for r in rows], c=color, alpha=.7, s=30)
    axs[1, 1].axhline(30, color="black", linestyle="--", linewidth=1)
    axs[1, 1].set(xticks=np.arange(4), xticklabels=labels, ylabel="Longest strict hold (s)",
                  title="All independent episodes; final validity also required")
    for ax in axs.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    fig.savefig(a.evaluation / "results_overview.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
