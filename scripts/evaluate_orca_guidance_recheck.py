"""Frozen initial/terminal evaluation on new states, using shared compiled physics."""

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--eval-start", type=int, default=6000)
    p.add_argument("--eval-count", type=int, default=32)
    args = p.parse_args()
    protocol, runs = read_runs(args.suite)
    if not protocol["chosen"]:
        raise ValueError("Heldout scoring is not hyperparameter selection")
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = list(range(args.eval_start, args.eval_start + args.eval_count))
    manifest = dict(suite=str(args.suite), seeds=seeds,
                    checkpoints="initial and terminal for every registered run; no best-of selection",
                    evaluation_reward="common shaped reward for diagnostics only, including terminal frame",
                    inference="deterministic, shared compiled full-action/residual evaluators",
                    protocol_sha256=hashlib.sha256((args.suite / "protocol.json").read_bytes()).hexdigest())
    (args.output / "protocol.json").write_text(json.dumps(manifest, indent=2))
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "artifacts/orca30/jax_cache"))
    net, _ = initialize_zero_mean(0, .7)
    envs = {arm: RewardAblationGrasp(arm, "shaped") for arm in ("pure", "residual")}
    evaluators = {arm: make_evaluator(env, net, seeds) for arm, env in envs.items()}
    records = []
    for run in runs:
        key = "residual" if run["arm"] == "residual" else "pure"
        for label, step in (("initial", 0), ("terminal", protocol["steps"])):
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
                seeds=seeds, hold_sec=holds.tolist(), success30=success.tolist()), indent=2))
            record = dict(run=run["run"], arm=run["arm"], reward=run["reward"],
                          seed=run["seed"], checkpoint=str(checkpoint), checkpoint_step=step,
                          checkpoint_sha256=hashlib.sha256(data).hexdigest(), label=label,
                          elapsed_sec=time.monotonic()-started, **row)
            records.append(record)
            (target / "summary.json").write_text(json.dumps(record, indent=2))
            with (args.output / "results.jsonl").open("a") as f:
                f.write(json.dumps(record)+"\n")
            print(json.dumps(record), flush=True)
    rng = np.random.default_rng(20261004)
    groups, paired = {}, {}
    for reward in protocol["rewards"]:
        for arm in protocol["arms"]:
            by_label = {}
            for label in ("initial", "terminal"):
                values = sorted([r for r in records if r["arm"] == arm and r["reward"] == reward
                                 and r["label"] == label], key=lambda r: r["seed"])
                by_label[label] = np.asarray([r["success30"] for r in values])
                groups[f"{reward}_{arm}_{label}"] = bootstrap_mean(by_label[label], rng)
            groups[f"{reward}_{arm}_gain"] = bootstrap_mean(by_label["terminal"]-by_label["initial"], rng)
        pure = groups[f"{reward}_pure_terminal"]["values"]
        for arm in ("bc", "residual"):
            guided = groups[f"{reward}_{arm}_terminal"]["values"]
            paired[f"{reward}_{arm}_minus_pure"] = bootstrap_mean(np.asarray(guided)-pure, rng)
    interaction = {}
    for arm in ("bc", "residual"):
        reduced = paired[f"hold_only_{arm}_minus_pure"]["values"]
        shaped = paired[f"shaped_{arm}_minus_pure"]["values"]
        interaction[arm] = bootstrap_mean(np.asarray(reduced)-shaped, rng)
    (args.output / "report.json").write_text(json.dumps(dict(
        complete=True, groups=groups, paired=paired, reward_interaction=interaction, records=records,
        note="five independent training seeds; 32 matched initial states per checkpoint; not 160 independent trainings"), indent=2))
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
