#!/usr/bin/env python3
"""Audit matching cap development settings and plot all evaluations."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--guided", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configs = [json.loads((d/"run_config.json").read_text()) for d in (args.scratch, args.guided)]
    for k in ("seed", "steps", "evals", "eval_count", "eval_start", "learning_rate", "init_std", "reward_version"):
        assert configs[0][k] == configs[1][k], k
    assert configs[0]["source_sha256"]["orca_cap_env.py"] == configs[1]["source_sha256"]["orca_cap_env.py"]
    for name in ("model.xml", "env_config.json"):
        assert (args.scratch/name).read_bytes() == (args.guided/name).read_bytes(), name
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    summary = dict(kind="one_training_seed_development_pair_not_confirmatory", common_settings_verified=True,
                   model_sha256=hashlib.sha256((args.scratch/"model.xml").read_bytes()).hexdigest(), arms={})
    for name, path, color in (("Scratch PPO", args.scratch, "#1976ad"),
                              ("Video grip BC + PPO", args.guided, "#b44722")):
        with (path/"audited_metrics.csv").open() as f:
            rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        assert rows[-1]["step"] >= configs[0]["steps"]
        best = max(rows, key=lambda r: (r["success"], r["mean_hold"], -r["step"]))
        x = np.array([r["step"] for r in rows])
        y = np.array([r["success"] for r in rows])
        summary["arms"][name] = dict(path=str(path), first=rows[0], last=rows[-1], best=best,
                                    max_success=float(y.max()), normalized_auc=float(np.trapezoid(y, x)/(x[-1]-x[0])),
                                    curve=rows)
        for axis, key, title in zip(axes, ("success", "mean_hold", "max_thread_error_m"),
                                    ("90 deg + 1 s success", "Mean valid hold (s)", "Peak thread error (mm)")):
            ys = [r[key]*(1000 if key == "max_thread_error_m" else 1) for r in rows]
            axis.plot(x/1000, ys, "o-", label=name, color=color)
            axis.set(xlabel="PPO control steps (thousands)", title=title)
            axis.grid(alpha=.25)
    axes[0].set_ylim(-.05, 1.05)
    axes[0].legend(fontsize=8)
    axes[2].axhline(1., color="#555555", linestyle="--", linewidth=1)
    fig.suptitle("Fixed bottle cap loosening: development pair, not a convergence claim")
    fig.savefig(args.output/"learning_curves.png", dpi=160)
    plt.close(fig)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({name: dict(best_step=r["best"]["step"], first=r["first"]["success"],
                                 last=r["last"]["success"], max=r["max_success"], auc=r["normalized_auc"])
                      for name, r in summary["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
