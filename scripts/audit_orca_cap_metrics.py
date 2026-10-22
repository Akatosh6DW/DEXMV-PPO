#!/usr/bin/env python3
"""Recount stored development trajectories; preserve original metrics.csv."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from orca_cap_metrics import summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", type=Path, nargs="+")
    args = p.parse_args()
    for run in args.runs:
        config = json.loads((run/"env_config.json").read_text())
        env = SimpleNamespace(dt=config["ctrl_dt"], _config=SimpleNamespace(**config))
        with (run/"metrics.csv").open() as f:
            rows = list(csv.DictReader(f))
        audited = []
        for row in rows:
            checkpoint = run/f"step_{int(row['step']):09d}"
            with np.load(checkpoint/"trajectories.npz") as z:
                trace = {k: z[k] for k in z.files}
            score, holds = summarize(trace, env)
            audited.append(dict(step=int(row["step"]), **score, elapsed_sec=float(row["elapsed_sec"])))
            (checkpoint/"audited_episodes.json").write_text(json.dumps(dict(
                criterion="whole_episode_no_forbidden_or_error_over_1mm", hold_sec=holds.tolist()), indent=2))
        with (run/"audited_metrics.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(audited[0]))
            writer.writeheader()
            writer.writerows(audited)
        best = max(audited, key=lambda r: (r["success"], r["mean_hold"], -r["step"]))
        print(json.dumps(dict(run=str(run), first=audited[0]["success"], last=audited[-1]["success"],
                              max=max(r["success"] for r in audited), best_step=best["step"], best=best), indent=2))


if __name__ == "__main__":
    main()
