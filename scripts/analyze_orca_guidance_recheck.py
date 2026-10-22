"""Analyze every predeclared run and choose noise using development data only."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_runs(suite):
    protocol = json.loads((suite / "protocol.json").read_text())
    rows = []
    for directory in sorted(suite.iterdir()):
        if not (directory / "summary.json").exists():
            continue
        config = json.loads((directory / "run_config.json").read_text())
        with (directory / "metrics.csv").open() as f:
            curve = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        rows.append(dict(run=directory.name, arm=config["arm"], reward=config["reward"],
                         seed=config["seed"], std=config["init_std"], curve=curve,
                         **json.loads((directory / "summary.json").read_text())))
    expected = len(protocol["seeds"]) * 6 * (1 if protocol["chosen"] else len(protocol["stds"]))
    if len(rows) != expected or not (suite / "completed.json").exists():
        raise ValueError(f"Incomplete suite: {len(rows)}/{expected}")
    return protocol, rows


def bootstrap_mean(values, rng):
    values = np.asarray(values)
    samples = rng.choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return dict(mean=float(values.mean()), sd=float(values.std(ddof=1)) if len(values) > 1 else None,
                ci95=np.quantile(samples, [.025, .975]).tolist(), values=values.tolist(), n=len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--select", type=Path)
    args = parser.parse_args()
    protocol, rows = read_runs(args.suite)
    result = dict(protocol=protocol, runs=rows)
    if args.select:
        if protocol["chosen"] or protocol["seeds"] != [91]:
            raise ValueError("Noise selection only allowed on P01 development seed91")
        chosen = {}
        for arm in protocol["arms"]:
            candidates = []
            for std in protocol["stds"]:
                group = [r for r in rows if r["arm"] == arm and r["std"] == std]
                if len(group) != 2:
                    raise ValueError("Incomplete reward pair")
                candidates.append(dict(std=std, mean_auc=float(np.mean([r["auc"] for r in group])),
                                       runs=[r["run"] for r in group]))
            winner = max(candidates, key=lambda r: (r["mean_auc"], r["std"]))
            chosen[arm] = dict(**winner, all_candidates=candidates,
                               selection_data="P01 only; not confirmatory/heldout")
        with args.select.open("x") as f:
            json.dump(chosen, f, indent=2)
        result["chosen"] = chosen
    else:
        rng = np.random.default_rng(20261004)
        result["groups"], result["paired"] = {}, {}
        for reward in protocol["rewards"]:
            for arm in protocol["arms"]:
                group = sorted([r for r in rows if r["reward"] == reward and r["arm"] == arm],
                               key=lambda r: r["seed"])
                result["groups"][f"{reward}_{arm}"] = {
                    metric: bootstrap_mean([r[metric] for r in group], rng)
                    for metric in ("first", "last", "auc", "gain_auc", "learning_gain")}
            pure = sorted([r for r in rows if r["reward"] == reward and r["arm"] == "pure"],
                          key=lambda r: r["seed"])
            for arm in ("bc", "residual"):
                guided = sorted([r for r in rows if r["reward"] == reward and r["arm"] == arm],
                                key=lambda r: r["seed"])
                result["paired"][f"{reward}_{arm}_minus_pure"] = {
                    metric: bootstrap_mean([a[metric] - b[metric] for a, b in zip(guided, pure)], rng)
                    for metric in ("last", "auc", "gain_auc", "learning_gain")}
        result["reward_interaction"] = {}
        for arm in ("bc", "residual"):
            result["reward_interaction"][arm] = {}
            for metric in ("last", "auc", "gain_auc", "learning_gain"):
                reduced = result["paired"][f"hold_only_{arm}_minus_pure"][metric]["values"]
                shaped = result["paired"][f"shaped_{arm}_minus_pure"][metric]["values"]
                result["reward_interaction"][arm][metric] = bootstrap_mean(np.asarray(reduced)-shaped, rng)
        result["interaction_note"] = "(guided - pure) under hold_only minus (guided - pure) under shaped; positive means larger relative benefit without geometric shaping"
        result["uncertainty_note"] = "paired training-seed bootstrap, not episode/frame pseudo-replication; 5-seed single-task coverage is uncertain"
    result["protocol_sha256"] = hashlib.sha256((args.suite / "protocol.json").read_bytes()).hexdigest()
    (args.suite / "analysis.json").write_text(json.dumps(result, indent=2))
    with (args.suite / "analysis.csv").open("w") as f:
        flat = [{k: v for k, v in r.items() if k != "curve"} for r in rows]
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({k: v for k, v in result.items() if k not in ("runs", "protocol")}, indent=2))


if __name__ == "__main__":
    main()
