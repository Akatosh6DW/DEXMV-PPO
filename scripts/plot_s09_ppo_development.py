"""Plot declared repeat chains without hiding late regressions or partial runs."""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from summarize_s09_ppo_improvement import BASE, origin_and_budget


def curve(names):
    points = []
    partial = False
    for name in names:
        matches = list(BASE.glob(name + "_*"))
        if not matches:
            partial = True
            continue
        if len(matches) != 1:
            raise ValueError("Ambiguous run prefix: " + name)
        run = matches[0]
        if not (run / "metrics.csv").exists():
            partial = True
            continue
        partial |= not (run / "summary.json").exists()
        with (run / "metrics.csv").open() as stream:
            for row in csv.DictReader(stream):
                step = int(row["step"])
                _, budget = origin_and_budget(run, step)
                points.append((budget, float(row["success30"]) * 100, float(row["hold_sec"])))
    return points, partial


def plot_group(chains, title, filename, budget_limit):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for col, (label, names) in enumerate(chains):
        points, partial = curve(names)
        axes[0, col].set_title(label + (" (partial)" if partial else " (complete)"))
        for row, (ylabel, index, bounds) in enumerate((
                ("Strict success (%)", 1, (0, 100)),
                ("Mean best valid hold (s)", 2, (0, 40)))):
            ax = axes[row, col]
            ax.set_ylabel(ylabel)
            ax.set_ylim(*bounds)
            ax.set_xlim(0, budget_limit / 1000)
            ax.grid(alpha=.2)
            ax.set_xlabel("Cumulative PPO interactions (thousands)")
            if points:
                ax.axhline(points[0][index], color="#777777", linestyle="--", label="Matched starting policy")
                ax.plot([p[0]/1000 for p in points], [p[index] for p in points],
                    color=("#087e8b", "#bb4430", "#407744")[col], marker="o", markersize=4,
                    label="Recorded development evaluations")
                if row == 0:
                    ax.legend(fontsize=8, loc="lower left")
            else:
                ax.text(.5, .5, "Not yet evaluated", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle(title + "\nSame 64 development cases; not heldout results; no curve smoothing", fontsize=14)
    output = BASE / "report" / filename
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


def main():
    plot_group([
        ("BC origin 13101 / PPO origin 14101", ["E19", "E24", "E55"]),
        ("BC origin 13102 / PPO origin 14102", ["E27", "E30", "E46"]),
        ("BC origin 13103 / PPO origin 14103", ["E29", "E32", "E56"])],
        "BC + PPO: matched slow-actor / fast-critic repeat chains", "bc_repeat_curves.png", 1310720)
    plot_group([
        ("PPO seed 14101", ["E43"]),
        ("PPO seed 14102", ["E52"]),
        ("PPO seed 14103", ["E54"])],
        "Residual PPO: 80-step fragments, fixed 40,960-interaction budget", "residual_repeat_curves.png", 40960)


if __name__ == "__main__":
    main()
