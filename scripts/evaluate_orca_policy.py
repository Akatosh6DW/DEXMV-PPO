#!/usr/bin/env python3
"""Reload an Orca policy and independently count strict continuous hold."""

import argparse
import json
from pathlib import Path
import pickle

import jax
from ml_collections import ConfigDict
import numpy as np

from orca_benchmark_env import OrcaBenchmark
from train_orca_comparison import initialize, make_evaluator


def longest_run(mask, dt=.05):
    best = count = 0
    for valid in mask:
        count = count + 1 if valid else 0
        best = max(best, count)
    return best * dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed-start", type=int, default=2000)
    p.add_argument("--count", type=int, default=32)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = ConfigDict(json.loads((args.checkpoint.parent / "env_config.json").read_text()))
    env = OrcaBenchmark(cfg)
    net, _ = initialize(env, 0)
    with (args.checkpoint / "params.pkl").open("rb") as f:
        params = pickle.load(f)
    seeds = list(range(args.seed_start, args.seed_start + args.count))
    trace = jax.device_get(make_evaluator(env, net, seeds)(params))
    finite = np.isfinite(trace["qpos"]).all(-1) & np.isfinite(trace["qvel"]).all(-1)
    valid = (trace["alive"] & finite & (trace["released"] > 0) &
             (trace["forbidden"] == 0) & (trace["thumb"] > 0) &
             (trace["contacts"] >= 2) & (trace["cube_z"] > trace["spawn_z"] - .01))
    holds = np.array([longest_run(mask) for mask in valid])
    np.testing.assert_allclose(holds, trace["hold"].max(axis=1), atol=1e-4)
    trace["independent_valid"] = valid
    np.savez_compressed(args.output / "trajectories.npz", **trace)
    report = dict(checkpoint=str(args.checkpoint.resolve()), seeds=seeds,
                  hold_sec=holds.tolist(), success30=float(np.mean(holds >= 30)),
                  mean_hold=float(holds.mean()), minimum_hold=float(holds.min()),
                  no_auto_reset=True, independent_counter_matches=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
