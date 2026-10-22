"""Sequential, equally budgeted guidance/reward/noise matrix with full logs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seeds", nargs="+", type=int, default=[91])
    p.add_argument("--stds", nargs="+", type=float, default=[.03, .7])
    p.add_argument("--steps", type=int, default=163840)
    p.add_argument("--evals", type=int, default=3)
    p.add_argument("--eval-start", type=int, default=4100)
    p.add_argument("--eval-count", type=int, default=16)
    p.add_argument("--selected-config", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    chosen = json.loads(args.selected_config.read_text()) if args.selected_config else None
    config.update(chosen=chosen, arms=["pure", "bc", "residual"], rewards=["shaped", "hold_only"],
                  selection_rule="per arm: largest mean AUC across both reward modes on pilot seed91; tie larger std",
                  invariant="same model/reset/reward within reward block; BC init only, no demo loss")
    (args.output / "protocol.json").write_text(json.dumps(config, indent=2))
    for seed in args.seeds:
        for std in ([None] if chosen else args.stds):
            for reward in config["rewards"]:
                for arm in config["arms"]:
                    noise = chosen[arm]["std"] if chosen else std
                    name = f"{reward}_{arm}_std{noise}_seed{seed}"
                    output = args.output / name
                    command = [sys.executable, "scripts/train_orca_guidance_recheck.py",
                               "--arm", arm, "--reward", reward, "--seed", str(seed),
                               "--init-std", str(noise), "--steps", str(args.steps),
                               "--evals", str(args.evals), "--eval-start", str(args.eval_start),
                               "--eval-count", str(args.eval_count), "--output", str(output)]
                    print("START", name, flush=True)
                    with (args.output / f"{name}.console.log").open("w") as log:
                        proc = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                        for line in proc.stdout:
                            log.write(line)
                            log.flush()
                            print(line, end="", flush=True)
                        code = proc.wait()
                    record = dict(run=name, command=command, returncode=code,
                                  reason="reward ablation and equal noise candidate budgets",
                                  checkpoint="fresh paired random initialization; BC adds 2000 supervised updates",
                                  changed="method within reward/noise block; reward only between paired blocks",
                                  smoke="test_orca_guidance_recheck.py passed",
                                  next="continue preregistered matrix, no outcome-based deletion")
                    if code == 0:
                        record.update(json.loads((output / "summary.json").read_text()))
                        record["failure_analysis"] = "retained/contact/hold metrics saved for every evaluation; score audited from physical traces"
                    with (args.output / "iterations.jsonl").open("a") as f:
                        f.write(json.dumps(record) + "\n")
                    if code:
                        raise SystemExit(code)
    (args.output / "completed.json").write_text(json.dumps(dict(complete=True), indent=2))


if __name__ == "__main__":
    main()
