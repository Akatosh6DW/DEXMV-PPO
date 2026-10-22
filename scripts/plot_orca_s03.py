"""S03 fixed-budget curves, held-out results and failure breakdown figures."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_orca_s03_confirmation import OUT

LABELS = dict(ppo_fixed="Fixed PPO", ppo_tuned="Tuned PPO", guided_tuned="Tuned BC + PPO",
              residual_tuned="Tuned residual", bc="BC only", zero="Reference only")
COLORS = dict(ppo_fixed="#92999e", ppo_tuned="#374f6c", guided_tuned="#bd6e29",
              residual_tuned="#16827a", bc="#9a8055", zero="#383838")


def main():
    report = OUT / "report"
    stats = json.loads((report / "statistics.json").read_text())
    if not stats["complete"]:
        raise ValueError("Final plots require the complete matrix")
    output = report / "figures"
    output.mkdir(exist_ok=True)
    with (report / "learning_curves.csv").open() as stream:
        curves = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for group in list(LABELS)[:4]:
        rows = [r for r in curves if r["stage"] == "confirmation" and r["group"] == group]
        steps = sorted({int(r["step"]) for r in rows})
        seeds = sorted({int(r["seed"]) for r in rows})
        values = np.array([[float(next(r["clean_success30"] for r in rows
                                       if int(r["step"]) == step and int(r["seed"]) == seed))
                            for step in steps] for seed in seeds])*100
        if len(seeds) != 10 or len(steps) != 9:
            raise ValueError("Incomplete learning curves")
        for y in values:
            ax.plot(steps, y, color=COLORS[group], alpha=.12, linewidth=.7)
        ax.plot(steps, values.mean(axis=0), "o-", markersize=3, color=COLORS[group], label=LABELS[group])
    ax.axvline(163840, color="#888888", linestyle="--", linewidth=1)
    ax.set(xlabel="Online control steps (excludes development search and BC)",
           ylabel="Development clean success (%)", ylim=(-2, 102),
           title="16 development starts; thin lines show all ten seeds")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "learning_curves.png", dpi=150)
    plt.close(fig)
    for budget in (163840, 655360):
        fig, axes = plt.subplots(2, 4, figsize=(19, 9), layout="constrained")
        conditions = ["B0_nominal", "B1_position", "B2_yaw", "B3_friction", "B4_mass", "B5_push", "X_cuboid", "X_sphere"]
        for ax, condition in zip(axes.flat, conditions):
            rows = [next(r for r in stats["aggregate"] if r["condition"] == condition
                         and r["group"] == group and r["budget"] == (0 if group in ("bc", "zero") else budget))
                    for group in LABELS]
            y = np.array([r["clean_success30"] for r in rows])*100
            low = np.maximum(0, y-np.array([r["ci_low"] for r in rows])*100)
            high = np.maximum(0, np.array([r["ci_high"] for r in rows])*100-y)
            ax.bar(np.arange(6), y, color=[COLORS[r["group"]] for r in rows], yerr=[low, high], capsize=2)
            ax.set_xticks(np.arange(6), list(LABELS.values()), rotation=40, ha="right", fontsize=7)
            ax.set(title=condition, ylim=(0, 105), ylabel="Clean 30s success (%)")
            ax.grid(axis="y", alpha=.15)
        fig.suptitle(f"Held-out cases at {budget} training steps; marginal crossed-bootstrap 95% intervals")
        fig.savefig(output / f"held_out_{budget}.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    rows = [r for r in stats["aggregate"] if r["condition"] == "B0_nominal" and r["budget"] in (0, 655360)]
    x = np.arange(len(rows))
    clean = np.array([r["clean_success30"] for r in rows])*100
    raw = np.array([r["raw_success30"] for r in rows])*100
    axes[0].bar(x, clean, color="#16827a", label="Clean success")
    axes[0].bar(x, raw-clean, bottom=clean, color="#daaa41", label="Raw success, dirty contact")
    axes[0].bar(x, 100-raw, bottom=raw, color="#a8adb4", label="Task failure")
    axes[0].set_xticks(x, [LABELS[r["group"]] for r in rows], rotation=35, ha="right", fontsize=8)
    axes[0].set(title="Nominal long-budget outcome decomposition", ylim=(0, 105), ylabel="Episodes (%)")
    axes[0].legend(fontsize=7)
    rows = [r for r in stats["aggregate"] if r["condition"] == "mechanism"]
    y = np.array([r["clean_success30"] for r in rows])*100
    axes[1].bar(np.arange(len(rows)), y, color=["#92999e", "#bd6e29", "#16827a", "#374f6c"],
                yerr=[np.maximum(0, y-np.array([r["ci_low"] for r in rows])*100),
                      np.maximum(0, np.array([r["ci_high"] for r in rows])*100-y)], capsize=3)
    axes[1].set_xticks(np.arange(len(rows)), [r["mode"].replace("_", " ") for r in rows], rotation=35, ha="right", fontsize=8)
    axes[1].set(title="Recorded residual versus state feedback", ylim=(0, 105), ylabel="Clean 30s success (%)")
    fig.savefig(output / "failures_and_mechanism.png", dpi=150)
    plt.close(fig)
    files = ["learning_curves.png", "held_out_163840.png", "held_out_655360.png", "failures_and_mechanism.png"]
    (output / "manifest.json").write_text(json.dumps(dict(complete=True, files=files,
        note="Curves are development data; bars are held-out test data. BC/reference controls are shared between budgets, not independent duplicates."), indent=2)+"\n")
    print(json.dumps(dict(plots=files)))


if __name__ == "__main__":
    main()
