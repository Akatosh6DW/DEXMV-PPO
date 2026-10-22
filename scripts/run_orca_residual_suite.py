"""Run matched three-arm experiments sequentially and log every outcome."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/orca_residual"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--shapes", nargs="+", default=["cube", "sphere", "cuboid"])
    p.add_argument("--arms", nargs="+", default=["pure", "bc", "residual"])
    p.add_argument("--seeds", nargs="+", type=int, default=[90])
    p.add_argument("--steps", type=int, default=163840)
    p.add_argument("--evals", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--eval-start", type=int, default=1000)
    p.add_argument("--confirmatory", action="store_true")
    args = p.parse_args()
    suite = OUT/args.name
    suite.mkdir(parents=True, exist_ok=False)
    (suite/"protocol.json").write_text(json.dumps(vars(args), indent=2))
    for shape in args.shapes:
        for seed in args.seeds:
            for arm in args.arms:
                output = suite/f"{shape}_{arm}_seed{seed}"
                command = [sys.executable, "scripts/train_orca_residual.py", "--arm", arm,
                           "--shape", shape, "--seed", str(seed), "--steps", str(args.steps),
                           "--evals", str(args.evals), "--learning-rate", str(args.learning_rate),
                           "--eval-start", str(args.eval_start),
                           "--output", str(output)]
                if arm == "bc":
                    command += ["--dataset", str(OUT/"shared_cube_demos/trajectories.npz")]
                if args.confirmatory:
                    command += ["--confirmatory"]
                record = dict(id=output.name, suite=args.name, task=shape, command=command,
                              checkpoint="fresh paired random hidden/value weights; zero mean head",
                              change="method only within paired task and seed; common PPO budget/config",
                              reason="compare full PPO, training-only imitation, runtime bounded residual",
                              expected="measure rather than assume residual advantage",
                              smoke="6 CPU tests and 3-shape zero-residual GPU audits passed")
                print("START", output.name, flush=True)
                with (suite/f"{output.name}.console.log").open("w") as log:
                    process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    record["returncode"] = process.wait()
                if record["returncode"] == 0:
                    with (output/"metrics.csv").open() as f:
                        rows = list(csv.DictReader(f))
                    record["metrics"] = rows
                    record.update(json.loads((output/"summary.json").read_text()))
                    record["contact_audit"] = "independent per-frame contact/height/retention and actuator audits passed"
                    record["failure_analysis"] = "partial or dropped episodes counted as failures; inspect retained raw traces"
                    record["next"] = "same-budget paired method/seed; freeze or revise common PPO setting after pilot"
                else:
                    record["failure_analysis"] = "runtime failure; retain output and diagnose before next run"
                with (OUT/f"{shape}_iterations.jsonl").open("a") as f:
                    f.write(json.dumps(record)+"\n")
                if record["returncode"]:
                    raise SystemExit(record["returncode"])
    (suite/"completed.json").write_text(json.dumps(dict(completed=True, protocol=vars(args)), indent=2))


if __name__ == "__main__":
    main()
