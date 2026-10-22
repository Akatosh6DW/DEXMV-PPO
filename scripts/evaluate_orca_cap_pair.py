#!/usr/bin/env python3
"""Reload selected and terminal cap policies on one common held-out set."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle

import jax
from ml_collections import config_dict
import numpy as np

from orca_cap_env import OrcaCap
from train_orca_cap import make_evaluator, summarize
from train_orca_comparison import network_factory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--guided", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", type=int, default=2000)
    p.add_argument("--count", type=int, default=16)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.scratch/"env_config.json").read_text())
    current_hash = hashlib.sha256(Path(__file__).with_name("orca_cap_env.py").read_bytes()).hexdigest()
    for path in (args.scratch, args.guided):
        assert json.loads((path/"env_config.json").read_text()) == config
        saved = json.loads((path/"run_config.json").read_text())
        assert saved["source_sha256"]["orca_cap_env.py"] == current_hash
        assert (path/"model.xml").read_bytes() == (args.scratch/"model.xml").read_bytes()
    env = OrcaCap(args.scratch/"model.xml", config_dict.ConfigDict(config))
    net = network_factory({"state": (95,), "privileged_state": (112,)}, env.action_size)
    seeds = list(range(args.start, args.start+args.count))
    evaluate = make_evaluator(env, net, seeds)
    reports = {}
    for arm, path in (("scratch", args.scratch), ("guided", args.guided)):
        with (path/"audited_metrics.csv").open() as f:
            rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        assert rows[-1]["step"] >= json.loads((path/"run_config.json").read_text())["steps"]
        best = max(rows, key=lambda r: (r["success"], r["mean_hold"], -r["step"]))
        for which, row in (("selected", best), ("terminal", rows[-1])):
            name = f"{arm}_{which}"
            checkpoint = path/f"step_{int(row['step']):09d}"/"params.pkl"
            with checkpoint.open("rb") as f:
                params = pickle.load(f)
            trace = jax.device_get(evaluate(params))
            summary, holds = summarize(trace, env)
            output = args.output/name
            output.mkdir()
            np.savez_compressed(output/"trajectories.npz", **trace)
            report = dict(backend="mjx_fresh_policy_rollouts", training_step=int(row["step"]), seeds=seeds,
                          checkpoint=str(checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                          hold_sec=holds.tolist(), summary=summary)
            (output/"report.json").write_text(json.dumps(report, indent=2))
            reports[name] = report
            (args.output/"summary.json").write_text(json.dumps(reports, indent=2))
            print(name, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
