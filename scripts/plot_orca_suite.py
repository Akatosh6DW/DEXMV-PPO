"""Plot existing S02 tables without selecting models or changing statistics."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = dict(ppo="PPO", guided="BC + PPO", bc="BC", residual="Residual PPO",
              linear="Linear + residual", unrefined="Unrefined + residual",
              zero_video="Reference only", zero_linear="Linear only",
              zero_unrefined="Unrefined only", half_residual="Half residual")
COLORS = dict(ppo="#677079", guided="#bf651f", bc="#a08546", residual="#16827a",
              linear="#556fb0", unrefined="#ad4b6b", zero_video="#414141",
              zero_linear="#7a87a5", zero_unrefined="#ad8c99", half_residual="#5a9d95")


def read_csv(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    report = root / "report"
    stats = json.loads((report / "statistics.json").read_text())
    status = ("Evaluation matrix complete" if stats["complete"] else
              f"INCOMPLETE evaluation matrix: {len(stats['missing'])} batches pending")
    output = report / "figures"
    output.mkdir(exist_ok=True)
    curves = read_csv(report / "learning_curves.csv")
    saved = []
    if curves:
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        for method in ("ppo", "guided", "residual", "linear", "unrefined"):
            subset = [r for r in curves if r["method"] == method]
            if not subset:
                continue
            steps = sorted({int(r["step"]) for r in subset})
            seeds = sorted({int(r["seed"]) for r in subset})
            data = np.array([[float(next(r["success30"] for r in subset
                                          if int(r["step"]) == t and int(r["seed"]) == s))
                              for t in steps] for s in seeds]) * 100
            for values in data:
                ax.plot(steps, values, color=COLORS[method], alpha=.12, linewidth=.8)
            ax.plot(steps, data.mean(axis=0), marker="o", markersize=3, color=COLORS[method],
                    label=f"{LABELS[method]} (n={len(seeds)})")
        ax.set(xlabel="Online training control steps", ylabel="Development raw success (%)", ylim=(-2, 102),
               title="Fixed checkpoints, 8 development starts; thin lines = individual seeds")
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
        fig.suptitle(status, fontsize=10)
        fig.savefig(output / "learning_curves.png", dpi=160)
        plt.close(fig)
        saved.append("learning_curves.png")
    aggregate = stats["aggregate"]
    if aggregate:
        fig, axes = plt.subplots(2, 3, figsize=(17, 10), layout="constrained")
        methods = ("ppo", "bc", "guided", "zero_video", "residual", "linear", "unrefined")
        benchmarks = json.loads((root / "protocol.json").read_text())["benchmark_names"]
        for ax, bench in zip(axes.flat, benchmarks):
            rows = [next((r for r in aggregate if r["benchmark"] == bench and r["method"] == m), None)
                    for m in methods]
            rows = [r for r in rows if r]
            x = np.arange(len(rows))
            values = np.array([r["clean_success30"]*100 for r in rows])
            lower = np.maximum(0, values - np.array([r["ci_low"]*100 for r in rows]))
            upper = np.maximum(0, np.array([r["ci_high"]*100 for r in rows]) - values)
            ax.bar(x, values, color=[COLORS[r["method"]] for r in rows], yerr=[lower, upper], capsize=3)
            ax.set_xticks(x, [LABELS[r["method"]] for r in rows], rotation=45, ha="right", fontsize=8)
            ax.set(title=bench, ylim=(0, 105), ylabel="Clean 30s success (%)")
            ax.grid(axis="y", alpha=.2)
            if not rows:
                ax.text(.5, .5, "No complete method groups yet\nNot a zero-success result",
                        ha="center", va="center", transform=ax.transAxes, fontsize=9)
        fig.suptitle(status + "\nFinal policies; marginal crossed-bootstrap 95% intervals, not simultaneous intervals")
        fig.savefig(output / "clean_success.png", dpi=150)
        plt.close(fig)
        saved.append("clean_success.png")
        nominal = [r for r in aggregate if r["benchmark"] == "B0_nominal"]
        if nominal:
            fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
            x = np.arange(len(nominal))
            clean = np.array([r["clean_success30"]*100 for r in nominal])
            raw = np.array([r["raw_success30"]*100 for r in nominal])
            ax.bar(x, clean, color="#16827a", label="Clean success")
            ax.bar(x, raw-clean, bottom=clean, color="#d8a345", label="Raw success, failed quality audit")
            ax.bar(x, 100-raw, bottom=raw, color="#b0b3b8", label="Task failure")
            ax.set_xticks(x, [LABELS[r["method"]] for r in nominal], rotation=35, ha="right", fontsize=8)
            ax.set(ylabel="Percentage of all test episodes", ylim=(0, 105), title="B0 outcome decomposition")
            ax.legend(fontsize=8)
            fig.suptitle(status, fontsize=10)
            fig.savefig(output / "nominal_outcomes.png", dpi=160)
            plt.close(fig)
            saved.append("nominal_outcomes.png")
    (output / "manifest.json").write_text(json.dumps(dict(complete=stats["complete"], files=saved,
        status=status,
        note="Curves use development raw success, not final clean success. Error bars are marginal intervals. Empty panels have no complete groups, not zero success."), indent=2)+"\n")
    print(json.dumps(dict(complete=stats["complete"], plots=saved)))


if __name__ == "__main__":
    main()
