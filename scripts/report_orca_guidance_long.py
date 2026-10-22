"""Quantitative tables and seed-level plots for the registered long experiment."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/orca_guidance_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_orca_guidance_recheck import bootstrap_mean, read_runs

ARMS = ("pure", "bc", "residual")
REWARDS = ("shaped", "hold_only")
NAMES = {"pure": "PPO from scratch", "bc": "BC init + PPO", "residual": "Video + residual PPO"}
COLORS = {"pure": "#176a9a", "bc": "#b65b10", "residual": "#247c4e"}


def confirmed_threshold(curve, threshold):
    """Time is the second confirming checkpoint, not a future-informed first hit."""
    for previous, current in zip(curve, curve[1:]):
        if previous["success30"] >= threshold and current["success30"] >= threshold:
            return int(current["step"])
    return None


def bootstrap_band(values, seed=20261004):
    values = np.asarray(values, float)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite seed-by-checkpoint matrix")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), (10000, len(values)))
    samples = values[indices].mean(axis=1)
    return values.mean(axis=0), np.quantile(samples, .025, axis=0), np.quantile(samples, .975, axis=0)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_runs(runs, reward, arm):
    return sorted([r for r in runs if r["reward"] == reward and r["arm"] == arm], key=lambda r: r["seed"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--nominal", type=Path, required=True)
    p.add_argument("--offset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol, runs = read_runs(args.suite)
    tests = {name: json.loads((path / "report.json").read_text())
             for name, path in (("nominal", args.nominal), ("offset", args.offset))}
    if any(not r["complete"] for r in tests.values()):
        raise ValueError("Incomplete heldout tests")
    args.output.mkdir(parents=True, exist_ok=False)
    x = np.array([r["step"] for r in runs[0]["curve"]])
    for run in runs:
        np.testing.assert_array_equal(x, [r["step"] for r in run["curve"]])
    rng = np.random.default_rng(20261004)
    rows, groups, thresholds = [], {}, {}
    for run in runs:
        row = {k: run[k] for k in ("run", "reward", "arm", "seed", "first", "last", "max",
                                  "best_step", "auc", "gain_auc", "learning_gain")}
        row.update(short_success=next(r["success30"] for r in run["curve"] if r["step"] == 655360),
                   confirm50=confirmed_threshold(run["curve"], .5),
                   confirm75=confirmed_threshold(run["curve"], .75),
                   elapsed_sec=run["curve"][-1]["elapsed_sec"],
                   bc_sec=json.loads((args.suite / run["run"] / "run_config.json").read_text())["bc_sec"])
        row["gain_from_short"] = row["last"]-row["short_success"]
        rows.append(row)
    write_csv(args.output / "training_runs.csv", rows)
    write_csv(args.output / "heldout_records.csv", [dict(condition=c, **r)
              for c, report in tests.items() for r in report["records"]])

    for gain in (False, True):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
        for ax, reward in zip(axes, REWARDS):
            for arm in ARMS:
                values = np.array([[r["success30"] for r in run["curve"]]
                                   for run in grouped_runs(runs, reward, arm)])
                if gain:
                    values = values-values[:, :1]
                mean, lo, hi = bootstrap_band(values)
                for value in values:
                    ax.plot(x/1e6, value, color=COLORS[arm], alpha=.15, lw=.8)
                ax.plot(x/1e6, mean, color=COLORS[arm], label=NAMES[arm], lw=2)
                ax.fill_between(x/1e6, lo, hi, color=COLORS[arm], alpha=.12)
            ax.axvline(.655360, color="#777777", ls=":", label="Previous budget")
            ax.set_title(reward)
            ax.set_xlabel("PPO environment steps (millions)")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Success minus initial" if gain else "30 s grasp success")
        if not gain:
            axes[0].set_ylim(-.02, 1.02)
        fig.suptitle("Five seeds; thin lines = runs, bands = pointwise 95% seed bootstrap")
        save(fig, args.output / ("learning_gain.png" if gain else "learning_curves.png"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for i, reward in enumerate(REWARDS):
        for j, threshold in enumerate((.5, .75)):
            ax = axes[i, j]
            for arm in ARMS:
                selected = grouped_runs(runs, reward, arm)
                hits = [confirmed_threshold(r["curve"], threshold) for r in selected]
                rate = [np.mean([v is not None and v <= step for v in hits]) for step in x]
                ax.step(x/1e6, rate, where="post", color=COLORS[arm], label=NAMES[arm])
                thresholds[f"{reward}_{arm}_{threshold}"] = dict(
                    confirming_steps=hits, censored=sum(v is None for v in hits), total=len(hits),
                    initially_above_seeds=[r["seed"] for r in selected if r["first"] >= threshold],
                    seeds_ended_below=[r["seed"] for r, hit in zip(selected, hits)
                                       if hit is not None and r["last"] < threshold])
            ax.set_title(f"{reward}: two evaluations >= {threshold:.0%}")
            ax.set_ylim(-.02, 1.02)
            ax.set_xlabel("PPO environment steps (millions)")
            ax.set_ylabel("Fraction of seeds confirmed")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle("Unreached runs remain censored; attainment is not permanent stability")
    save(fig, args.output / "threshold_attainment.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, reward in enumerate(REWARDS):
        for j, metric in enumerate(("auc", "gain_auc")):
            ax = axes[i, j]
            for k, arm in enumerate(ARMS):
                values = [r[metric] for r in rows if r["reward"] == reward and r["arm"] == arm]
                stat = bootstrap_mean(values, rng)
                groups[f"{reward}_{arm}_{metric}"] = stat
                ax.scatter(k+np.linspace(-.08, .08, len(values)), values, color=COLORS[arm], alpha=.5)
                ax.errorbar(k, stat["mean"], yerr=np.array([[stat["mean"]-stat["ci95"][0]],
                            [stat["ci95"][1]-stat["mean"]]]), color=COLORS[arm], fmt="D", capsize=5)
            ax.axhline(0, color="#777777", lw=.6)
            ax.set_xticks(range(3), ["PPO", "BC + PPO", "Residual PPO"])
            ax.set_title(f"{reward}: {metric}")
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Area under success curve; gain AUC subtracts initial competence")
    save(fig, args.output / "auc.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for i, reward in enumerate(REWARDS):
        for j, condition in enumerate(("nominal", "offset")):
            ax = axes[i, j]
            labels = ["initial", "short", "terminal"] if condition == "nominal" else ["initial", "terminal"]
            for arm in ARMS:
                values = np.array([tests[condition]["groups"][f"{reward}_{arm}_{label}"]["values"]
                                   for label in labels]).T
                mean, lo, hi = bootstrap_band(values)
                xx = np.arange(len(labels))
                ax.errorbar(xx, mean, yerr=np.stack([mean-lo, hi-mean]), fmt="o-",
                            color=COLORS[arm], label=NAMES[arm], capsize=4)
            ax.set_xticks(range(len(labels)), labels)
            ax.set_title(f"{reward}: {condition} (XY +/- {2 if condition == 'nominal' else 8} mm)")
            ax.set_ylim(-.02, 1.02)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("Heldout success, 64 initial states/model")
    axes[1, 0].set_ylabel("Heldout success, 64 initial states/model")
    fig.suptitle("Fixed checkpoints only; intervals resample 5 training seeds")
    save(fig, args.output / "heldout_success.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    robustness = {}
    for ax, reward in zip(axes, REWARDS):
        for k, arm in enumerate(ARMS):
            nominal = np.array(tests["nominal"]["groups"][f"{reward}_{arm}_terminal"]["values"])
            offset = np.array(tests["offset"]["groups"][f"{reward}_{arm}_terminal"]["values"])
            stat = bootstrap_mean(offset-nominal, rng)
            robustness[f"{reward}_{arm}"] = stat
            ax.scatter(k+np.linspace(-.08, .08, len(nominal)), offset-nominal, color=COLORS[arm], alpha=.5)
            ax.errorbar(k, stat["mean"], yerr=[[stat["mean"]-stat["ci95"][0]],
                         [stat["ci95"][1]-stat["mean"]]], fmt="D", color=COLORS[arm], capsize=5)
        ax.axhline(0, color="#777777", lw=.8)
        ax.set_title(reward)
        ax.set_xticks(range(3), ["PPO", "BC + PPO", "Residual PPO"])
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Success at 8 mm minus success at 2 mm")
    fig.suptitle("Distribution shift sensitivity; paired by trained model, not initial state")
    save(fig, args.output / "robustness.png")
    diagnostic_rows = []
    for condition, test in tests.items():
        for reward in REWARDS:
            for arm in ARMS:
                records = sorted([r for r in test["records"] if (r["reward"], r["arm"], r["label"]) ==
                                  (reward, arm, "terminal")], key=lambda r: r["seed"])
                for metric in ("mean_hold", "final_hold30", "retained", "active_residual_rms_rad",
                               "active_residual_saturation"):
                    stat = bootstrap_mean([r[metric] for r in records], rng)
                    diagnostic_rows.append(dict(condition=condition, reward=reward, arm=arm, metric=metric,
                                                mean=stat["mean"], ci_low=stat["ci95"][0],
                                                ci_high=stat["ci95"][1], values=json.dumps(stat["values"])))
    write_csv(args.output / "terminal_diagnostics.csv", diagnostic_rows)
    budget_gains = []
    for condition, test in tests.items():
        for reward in REWARDS:
            for arm in ARMS:
                prefix = f"{reward}_{arm}_"
                initial = np.array(test["groups"][prefix+"initial"]["values"])
                for label in ("short", "terminal"):
                    if prefix+label not in test["groups"]:
                        continue
                    values = np.array(test["groups"][prefix+label]["values"])-initial
                    stat = bootstrap_mean(values, rng)
                    budget_gains.append(dict(condition=condition, reward=reward, arm=arm, label=label,
                                             mean=stat["mean"], ci_low=stat["ci95"][0], ci_high=stat["ci95"][1],
                                             improved=int(np.sum(values > 0)), worsened=int(np.sum(values < 0)),
                                             unchanged=int(np.sum(values == 0)), values=json.dumps(stat["values"])))
    write_csv(args.output / "heldout_gains.csv", budget_gains)
    costs = dict(ppo_training_steps=len(runs)*protocol["steps"],
                 script_elapsed_sec=sum(r["elapsed_sec"] for r in rows),
                 bc_sec=sum(r["bc_sec"] for r in rows),
                 heldout_elapsed_sec=sum(r["elapsed_sec"] for t in tests.values() for r in t["records"]),
                 prior_preparation_cost="not measured; not zero")
    result = dict(complete=True, protocol=protocol, training=rows, auc=groups, thresholds=thresholds,
                  robustness=robustness, terminal_diagnostics=diagnostic_rows, heldout_gains=budget_gains, costs=costs,
                  reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  uncertainty="5 training seeds; pointwise bootstrap, no multiple-comparison correction")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    md = ["# Long-Study Quantitative Results", "", "All five training seeds; no best-checkpoint selection.", "",
          "| Condition | Reward | Method | Initial | Short | Terminal | Terminal - Initial |",
          "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for condition, test in tests.items():
        for reward in REWARDS:
            for arm in ARMS:
                stats = test["groups"]
                prefix = f"{reward}_{arm}_"
                short = f"{stats[prefix+'short']['mean']:.2%}" if prefix+"short" in stats else "not tested"
                md.append(f"| {condition} | {reward} | {arm} | {stats[prefix+'initial']['mean']:.2%} | "
                          f"{short} | {stats[prefix+'terminal']['mean']:.2%} | "
                          f"{100*stats[prefix+'gain_from_initial']['mean']:+.2f} pp |")
    md.extend(["", "## Fixed-Budget Paired Gains", "",
               "All registered checkpoints minus their own initial model; intervals resample training seeds.", "",
               "| Condition | Reward | Method | Checkpoint | Gain (pp) | 95% interval (pp) | Improved / unchanged / worsened |",
               "| --- | --- | --- | --- | ---: | --- | --- |"])
    for row in budget_gains:
        md.append(f"| {row['condition']} | {row['reward']} | {row['arm']} | {row['label']} | "
                  f"{100*row['mean']:+.2f} | [{100*row['ci_low']:+.2f}, {100*row['ci_high']:+.2f}] | "
                  f"{row['improved']} / {row['unchanged']} / {row['worsened']} |")
    md.extend(["", "## Attainment and Later Regression", "",
               "A confirmed threshold crossing can be followed by deterioration; censored runs are not assigned fake times.", "",
               "| Reward | Method | Threshold | Initially above / 5 | Confirmed / 5 | Ended below after confirmation | Confirming steps by seed |",
               "| --- | --- | ---: | ---: | ---: | ---: | --- |"])
    for reward in REWARDS:
        for arm in ARMS:
            for threshold in (.5, .75):
                entry = thresholds[f"{reward}_{arm}_{threshold}"]
                times = ", ".join("censored" if v is None else str(v) for v in entry["confirming_steps"])
                md.append(f"| {reward} | {arm} | {threshold:.0%} | {len(entry['initially_above_seeds'])}/5 | {5-entry['censored']}/5 | "
                          f"{len(entry['seeds_ended_below'])} | {times} |")
    md.extend(["", "## Terminal Diagnostics", "",
               "Longest valid hold is not necessarily the final uninterrupted hold; alive is not necessarily a clean grasp.", "",
               "| Condition | Reward | Method | Mean longest hold (s) | Final hold >= 30s | Alive at end | Residual RMS (rad) | Saturation |",
               "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for condition in tests:
        for reward in REWARDS:
            for arm in ARMS:
                values = {r["metric"]: r["mean"] for r in diagnostic_rows if
                          (r["condition"], r["reward"], r["arm"]) == (condition, reward, arm)}
                md.append(f"| {condition} | {reward} | {arm} | {values['mean_hold']:.3f} | "
                          f"{values['final_hold30']:.2%} | {values['retained']:.2%} | "
                          f"{values['active_residual_rms_rad']:.5f} | {values['active_residual_saturation']:.2%} |")
    for filename in ("learning_curves", "learning_gain", "threshold_attainment", "auc", "heldout_success", "robustness"):
        md.extend(["", f"![{filename}]({filename}.png)"])
    md.extend(["", "Raw seed values and intervals: summary.json and the heldout report.json files.",
               "BC, demonstration preparation, validation and testing are extra to PPO environment steps."])
    (args.output / "results.md").write_text("\n".join(md)+"\n")
    print(json.dumps(dict(status="DONE", output=str(args.output), **costs)))


if __name__ == "__main__":
    main()
