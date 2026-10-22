#!/usr/bin/env python3
"""Describe paired training curves without inventing censored convergence."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def summarize(path):
    with (path / "metrics.csv").open() as f:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    step = np.array([r["step"] for r in rows])
    success = np.array([r["success30"] for r in rows])
    reached = np.flatnonzero((success[:-1] >= .8) & (success[1:] >= .8))
    onset = int(step[reached[0]]) if reached.size else None
    confirmed = int(step[reached[0] + 1]) if reached.size else None
    return dict(path=str(path), first=float(success[0]), last=float(success[-1]),
                max=float(success.max()), best_step=int(step[success.argmax()]),
                convergence_onset_step=onset, convergence_confirmed_step=confirmed,
                censored=onset is None, last_step=int(step[-1]),
                normalized_auc=float(np.trapezoid(success, step) / step[-1]) if step[-1] else 0,
                rows=rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scratch", type=Path, nargs="+", required=True)
    p.add_argument("--guided", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if len(args.scratch) != len(args.guided):
        p.error("Use equal numbers of paired training runs")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = {"scratch": [], "guided": []}
    for scratch, guided in zip(args.scratch, args.guided):
        configs = [json.loads((p / "run_config.json").read_text()) for p in (scratch, guided)]
        for key in ("seed", "steps", "evals", "eval_start", "eval_count", "learning_rate", "init_std", "model_sha256", "source_sha256"):
            if configs[0].get(key) != configs[1].get(key):
                raise ValueError(f"Unmatched {key} in {scratch} vs {guided}")
        if json.loads((scratch / "env_config.json").read_text()) != json.loads((guided / "env_config.json").read_text()):
            raise ValueError("Environment configurations differ")
        summaries["scratch"].append(summarize(scratch))
        summaries["guided"].append(summarize(guided))
    differences = np.array([b["normalized_auc"] - a["normalized_auc"]
                            for a, b in zip(summaries["scratch"], summaries["guided"])])
    rng = np.random.default_rng(42)
    boot = rng.choice(differences, (10000, len(differences))).mean(axis=1)
    summaries["paired_auc_difference"] = differences.tolist()
    summaries["mean_auc_difference"] = float(differences.mean())
    summaries["bootstrap_mean_difference_95pct"] = np.quantile(boot, [.025, .975]).tolist() if len(differences) > 1 else None
    summaries["warning"] = "Uncertainty is across training seeds, not episodes; development results are not confirmatory evidence."
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for arm, color in (("scratch", "#277DA1"), ("guided", "#D65F43")):
        for i, run in enumerate(summaries[arm]):
            ax.plot([r["step"] for r in run["rows"]], [r["success30"] for r in run["rows"]],
                    color=color, alpha=.7, marker="o", label=arm if i == 0 else None)
    ax.set(xlabel="PPO environment steps (demo preparation excluded)",
           ylabel="Continuous 30 s success fraction", ylim=(-.03, 1.03))
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / "learning_curves.png", dpi=150)
    plt.close(fig)
    print(json.dumps({k: v for k, v in summaries.items() if k not in ("scratch", "guided")}, indent=2))


if __name__ == "__main__":
    main()
