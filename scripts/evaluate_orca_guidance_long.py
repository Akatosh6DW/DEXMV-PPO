"""Predeclared fixed-budget evaluations for the continuous long study."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import jax
import numpy as np

from analyze_orca_guidance_recheck import bootstrap_mean, read_runs
from orca_guidance_recheck import ROOT, RewardAblationGrasp, make_evaluator, score
from train_orca_residual import initialize_zero_mean


def condition_config(condition):
    if condition == "nominal":
        return dict(jitter=.002, seeds=list(range(7000, 7064)),
                    checkpoints={"initial": 0, "short": 655360, "terminal": 2621440})
    if condition == "offset":
        return dict(jitter=.008, seeds=list(range(8000, 8064)),
                    checkpoints={"initial": 0, "terminal": 2621440})
    raise ValueError(condition)


def aggregate(records, protocol, labels):
    rng = np.random.default_rng(20261004)
    groups, paired = {}, {}
    for reward in protocol["rewards"]:
        for arm in protocol["arms"]:
            by_label = {}
            for label in labels:
                rows = sorted([r for r in records if (r["reward"], r["arm"], r["label"]) ==
                               (reward, arm, label)], key=lambda r: r["seed"])
                if [r["seed"] for r in rows] != sorted(protocol["seeds"]):
                    raise ValueError("Missing or duplicate training seeds")
                by_label[label] = np.array([r["success30"] for r in rows])
                groups[f"{reward}_{arm}_{label}"] = bootstrap_mean(by_label[label], rng)
            for base in ("initial", "short"):
                if base in by_label:
                    groups[f"{reward}_{arm}_gain_from_{base}"] = bootstrap_mean(
                        by_label["terminal"] - by_label[base], rng)
        for label in labels:
            pure = np.array(groups[f"{reward}_pure_{label}"]["values"])
            for arm in ("bc", "residual"):
                guided = np.array(groups[f"{reward}_{arm}_{label}"]["values"])
                paired[f"{reward}_{arm}_minus_pure_{label}"] = bootstrap_mean(guided-pure, rng)
    return dict(groups=groups, paired=paired)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--condition", required=True, choices=("nominal", "offset"))
    args = p.parse_args()
    protocol, runs = read_runs(args.suite)
    if protocol["steps"] != 2621440 or protocol["seeds"] != [20, 21, 22, 23, 24]:
        raise ValueError("Not the registered long-study matrix")
    cfg = condition_config(args.condition)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(suite=str(args.suite), condition=args.condition, **cfg,
                    inference="deterministic; common compiled graph per action interface",
                    evaluation_reward="shaped diagnostic return; success scored from physical state",
                    selection="fixed initial/short/terminal budgets, never best checkpoint",
                    protocol_sha256=hashlib.sha256((args.suite / "protocol.json").read_bytes()).hexdigest(),
                    evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / "protocol.json").write_text(json.dumps(manifest, indent=2))
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "artifacts/orca30/jax_cache"))
    net, _ = initialize_zero_mean(0, .7)
    envs = {arm: RewardAblationGrasp(arm, "shaped") for arm in ("pure", "residual")}
    for env in envs.values():
        env._config.spawn_config.cube_jitter = [cfg["jitter"], cfg["jitter"], 0.]
    evaluators = {arm: make_evaluator(env, net, cfg["seeds"]) for arm, env in envs.items()}
    records = []
    for run in runs:
        key = "residual" if run["arm"] == "residual" else "pure"
        for label, step in cfg["checkpoints"].items():
            checkpoint = args.suite / run["run"] / f"step_{step:09d}" / "params.pkl"
            data = checkpoint.read_bytes()
            params = pickle.loads(data)
            if not all(np.isfinite(v).all() for v in jax.tree.leaves(params)):
                raise ValueError("Nonfinite checkpoint")
            started = time.monotonic()
            trace = jax.device_get(evaluators[key](params))
            if not all(np.isfinite(v).all() for v in trace.values()):
                raise ValueError("Nonfinite physical trace")
            row, holds, success = score(trace, envs[key])
            target = args.output / f"{run['run']}_{label}"
            target.mkdir()
            np.savez_compressed(target / "trajectories.npz", **trace)
            (target / "episodes.json").write_text(json.dumps(dict(
                seeds=cfg["seeds"], hold_sec=holds.tolist(), success30=success.tolist()), indent=2))
            record = dict(run=run["run"], arm=run["arm"], reward=run["reward"], seed=run["seed"],
                          checkpoint=str(checkpoint), checkpoint_step=step, label=label,
                          checkpoint_sha256=hashlib.sha256(data).hexdigest(),
                          elapsed_sec=time.monotonic()-started, **row)
            records.append(record)
            (target / "summary.json").write_text(json.dumps(record, indent=2))
            with (args.output / "results.jsonl").open("a") as f:
                f.write(json.dumps(record)+"\n")
            print(json.dumps(record), flush=True)
    report = dict(complete=True, condition=args.condition, records=records,
                  **aggregate(records, protocol, cfg["checkpoints"]),
                  note="5 independent training seeds; 64 matched states, not 320 independent trainings")
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
