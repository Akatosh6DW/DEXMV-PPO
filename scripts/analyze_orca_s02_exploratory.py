"""Post-hoc exploratory analysis of finished S02 tables; reads only, never changes frozen results.

Everything here was computed after the S02 test results were seen. It is for planning the next
preregistered experiment (seed counts, test-state counts, candidate claims), not for confirmatory claims.
"""

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

BENCHMARKS = ("B0_nominal", "B1_position", "B2_yaw", "B3_friction", "B4_mass", "B5_push")
LEARNED = ("ppo", "bc", "guided", "residual", "linear", "unrefined")
LABELS = dict(ppo="PPO", guided="BC + PPO", residual="Residual PPO")
# Validated categorical slots 1..3 (all-pairs CVD-safe); x-position and tick labels carry identity too.
COLORS = dict(ppo="#2a78d6", guided="#eb6834", residual="#1baf7a")
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def seed_values(rows, benchmark, method):
    subset = sorted((r for r in rows if r["benchmark"] == benchmark and r["method"] == method),
                    key=lambda r: int(r["seed"]))
    return np.array([float(r["clean_success30"]) * 100 for r in subset])


def sign_flip_p(diff, rng, draws=4000):
    n = len(diff)
    if n <= 16:
        signs = np.array(list(itertools.product((-1, 1), repeat=n)))
        extra = 0
    else:
        signs = rng.choice((-1, 1), size=(draws, n))
        extra = 1
    observed = abs(diff.mean())
    flipped = np.abs((signs * diff).mean(axis=1))
    return (np.sum(flipped >= observed - 1e-12) + extra) / (len(flipped) + extra)


def reliability(rows):
    table = []
    for benchmark in BENCHMARKS:
        residual = seed_values(rows, benchmark, "residual")
        for method in LEARNED:
            x = seed_values(rows, benchmark, method)
            entry = dict(benchmark=benchmark, method=method, seeds=len(x), mean=x.mean(),
                         sd=x.std(ddof=1), min=x.min(), median=np.median(x),
                         iqm=stats.trim_mean(x, .25), seeds_below_50=int((x < 50).sum()))
            if method != "residual":
                entry["residual_prob_improvement"] = float(np.mean(
                    [(a > b) + .5 * (a == b) for a in residual for b in x]))
                entry["brown_forsythe_p"] = float(stats.levene(residual, x, center="median").pvalue)
            table.append(entry)
    return table


def seed_power(rows, rng, sims):
    """Plug-in power for the B0 residual-vs-PPO sign-flip test at new, equal seed counts."""
    residual = seed_values(rows, "B0_nominal", "residual")
    ppo = seed_values(rows, "B0_nominal", "ppo")
    result = []
    for n in (10, 15, 20, 30):
        pvalues = np.array([sign_flip_p(rng.choice(residual, n) - rng.choice(ppo, n), rng)
                            for _ in range(sims)])
        result.append(dict(seeds_per_method=n, power_alpha_0_0167=float(np.mean(pvalues < .05 / 3)),
                           power_alpha_0_01=float(np.mean(pvalues < .01))))
    return result


def state_precision(episodes):
    """Split residual-minus-reference uncertainty into test-state and training-seed parts."""
    outcome = {}
    for r in episodes:
        if r["method"] in ("residual", "zero_video"):
            key = (r["benchmark"], r["method"], int(r["train_seed"]), int(r["seed"]))
            outcome[key] = 100. if r["clean_success30"] == "True" else 0.
    result = []
    for benchmark in BENCHMARKS:
        states = sorted({k[3] for k in outcome if k[0] == benchmark and k[1] == "zero_video"})
        seeds = sorted({k[2] for k in outcome if k[0] == benchmark and k[1] == "residual"})
        residual = np.array([[outcome[(benchmark, "residual", s, t)] for s in seeds] for t in states])
        reference = np.array([outcome[(benchmark, "zero_video", 1001, t)] for t in states])
        per_state = residual.mean(axis=1) - reference
        per_seed = residual.mean(axis=0) - reference.mean()
        sd_state = per_state.std(ddof=1)
        se_seed = per_seed.std(ddof=1) / np.sqrt(len(seeds))
        entry = dict(benchmark=benchmark, difference_pp=per_state.mean(),
                     se_test_states=sd_state / np.sqrt(len(states)), se_training_seeds=se_seed)
        for count in (200, 1000, 2000):
            entry[f"approx_total_se_{count}_states"] = float(np.sqrt(sd_state ** 2 / count + se_seed ** 2))
        result.append(entry)
    return result


def plot(rows, aggregate, path):
    reference = {r["benchmark"]: float(r["clean_success30"]) * 100 for r in aggregate
                 if r["method"] == "zero_video"}
    methods = ("ppo", "guided", "residual")
    jitter = np.random.default_rng(7).uniform(-.16, .16, size=10)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharey=True, layout="constrained")
    for ax, benchmark in zip(axes.flat, BENCHMARKS):
        ax.axhline(reference[benchmark], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        for i, method in enumerate(methods):
            x = seed_values(rows, benchmark, method)
            ax.scatter(i + jitter[:len(x)], x, s=34, color=COLORS[method], edgecolor="#fcfcfb",
                       linewidth=1.2, zorder=3)
            ax.plot([i - .26, i + .26], [x.mean()] * 2, color=INK, linewidth=2, zorder=4)
        ax.set_title(benchmark, fontsize=10, color=INK)
        ax.set_xticks(range(len(methods)), [LABELS[m] for m in methods], fontsize=8.5)
        ax.set_xlim(-.55, 2.55)
        ax.set_ylim(-4, 104)
        ax.grid(axis="y", color=GRID, linewidth=.6)
        ax.tick_params(colors=MUTED, labelcolor="#52514e", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
    for ax in axes[:, 0]:
        ax.set_ylabel("Clean 30 s success per training seed (%)", fontsize=8.5, color="#52514e")
    handles = [plt.Line2D([], [], color=INK, linewidth=2, label="Mean of 10 training seeds"),
               plt.Line2D([], [], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
                          label="Reference only: corrected video reference, zero residual (deterministic)")]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False, fontsize=8.5)
    fig.suptitle("S02 test success per training seed (each dot = one model on the same 200 starts)\n"
                 "Post-hoc exploratory view of frozen results", fontsize=9.5, color="#52514e")
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sims", type=int, default=1000)
    args = p.parse_args()
    report = args.root.resolve() / "report"
    statistics = json.loads((report / "statistics.json").read_text())
    if not statistics["complete"]:
        raise SystemExit("S02 statistics are incomplete; refusing to analyse partial tables")
    rows = read_csv(report / "per_seed.csv")
    aggregate = read_csv(report / "aggregate.csv")
    episodes = read_csv(report / "episodes.csv")
    rng = np.random.default_rng(20261008)
    args.output.mkdir(parents=True, exist_ok=True)

    table = reliability(rows)
    with (args.output / "seed_reliability.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in table for k in r)))
        writer.writeheader()
        writer.writerows(table)
    summary = dict(
        status="post-hoc exploratory; not part of the S02 preregistered hypothesis families",
        seed_power_b0_residual_vs_ppo=seed_power(rows, rng, args.sims),
        residual_vs_reference_precision=state_precision(episodes),
        power_method="bootstrap seeds from observed S02 per-seed results; two-sided sign-flip test; "
                     "plug-in estimate, optimistic because it treats 10 observed seeds as the population",
    )
    (args.output / "planning_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    plot(rows, aggregate, args.output / "per_seed_success.png")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
