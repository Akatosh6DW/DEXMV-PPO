#!/usr/bin/env python3
"""Separate paired curves and a seed-level aggregate for readable inspection."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    base = Path(__file__).resolve().parents[1] / "artifacts/orca30/confirmatory_f01/analysis"
    summary = json.loads((base / "summary.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    steps = np.array([r["step"] for r in summary["scratch"][0]["rows"]]) / 1e6
    boot_indices = np.random.default_rng(42).integers(0, 5, size=(10000, 5))
    for arm, color, label in (("scratch", "#277DA1", "Scratch PPO"),
                              ("guided", "#D65F43", "Demonstration-guided PPO")):
        curves = np.array([[r["success30"] for r in run["rows"]] for run in summary[arm]])
        for seed, curve in enumerate(curves):
            axes.flat[seed].plot(steps, curve, color=color, lw=1.5, marker=".", label=label)
            axes.flat[seed].set_title(f"Training seed {seed}")
        lower, upper = np.quantile(curves[boot_indices].mean(axis=1), [.025, .975], axis=0)
        axes.flat[5].plot(steps, curves.mean(axis=0), color=color, lw=2, label=label)
        axes.flat[5].fill_between(steps, lower, upper, color=color, alpha=.16)
    axes.flat[5].set_title("Mean + pointwise 95% seed bootstrap")
    for ax in axes.flat:
        ax.set_ylim(-.03, 1.03)
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8, loc="lower right")
    fig.supxlabel("PPO environment steps (millions; demonstration preparation excluded)")
    fig.supylabel("Continuous 30-second success fraction")
    fig.tight_layout()
    fig.savefig(base / "paired_and_mean_curves.png", dpi=160)
    plt.close(fig)
    print(base / "paired_and_mean_curves.png")


if __name__ == "__main__":
    main()
