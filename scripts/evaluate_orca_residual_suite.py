"""Held-out evaluation of selected and terminal checkpoints with source checks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def verify_sources(run, meta):
    for name, expected in meta["source_sha256"].items():
        actual = hashlib.sha256((ROOT/"scripts"/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Source changed since training: {name}; do not silently evaluate different code")
    if meta["precision"] != "highest":
        raise ValueError("Training precision was not fixed")
    if meta["arm"] == "residual":
        reference = Path(meta["reference"])
        if hashlib.sha256(reference.read_bytes()).hexdigest() != meta["reference_sha256"]:
            raise ValueError(f"Reference changed since training: {run}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("suite", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--eval-start", type=int, default=2000)
    p.add_argument("--eval-count", type=int, default=32)
    args = p.parse_args()
    analysis = json.loads((args.suite/"analysis.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for record in analysis["runs"]:
        run = Path(record["run"])
        meta = json.loads((run/"run_config.json").read_text())
        verify_sources(run, meta)
        heldout = set(range(args.eval_start, args.eval_start+args.eval_count))
        validation = set(range(meta["eval_start"], meta["eval_start"]+meta["eval_count"]))
        if heldout & validation:
            raise ValueError("Held-out seeds overlap checkpoint-selection seeds")
        if meta["arm"] == "bc":
            source = Path(meta["dataset_provenance"]["source"])
            demo_seeds = json.loads((source.parent/"episodes.json").read_text())["seeds"]
            if heldout & set(demo_seeds):
                raise ValueError("Held-out seeds overlap demonstration-generation seeds")
        for kind, step in (("selected", record["best_step"]), ("terminal", record["actual_steps"])):
            output = args.output/f"{run.name}_{kind}"
            command = [sys.executable, "scripts/train_orca_residual.py", "--arm", meta["arm"],
                       "--shape", meta["shape"], "--seed", str(meta["seed"]),
                       "--bound", str(meta["bound"]), "--reference", str(meta["reference"]),
                       "--checkpoint", str(run/f"step_{step:09d}/params.pkl"),
                       "--eval-start", str(args.eval_start), "--eval-count", str(args.eval_count),
                       "--output", str(output), "--steps", "0", "--confirmatory"]
            print("EVALUATE", output.name, flush=True)
            subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)
            evaluation = json.loads((output/"run_config.json").read_text())
            if evaluation["model_sha256"] != meta["model_sha256"]:
                raise ValueError("Evaluation model differs from training")
            if (output/"env_config.json").read_bytes() != (run/"env_config.json").read_bytes():
                raise ValueError("Evaluation environment differs from training")
            episodes = json.loads((output/"step_000000000/episodes.json").read_text())
            row = dict(task=record["task"], arm=record["arm"], seed=record["seed"], kind=kind,
                       checkpoint_step=step, evaluation=str(output), **episodes)
            rows.append(row)
            (args.output/"results.json").write_text(json.dumps(rows, indent=2))
    (args.output/"completed.json").write_text(json.dumps(dict(completed=True, runs=len(rows)), indent=2))


if __name__ == "__main__":
    main()
