"""Prior-only held-out baselines and same-checkpoint cube repeatability checks."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from evaluate_orca_residual_suite import ROOT, verify_sources


def main():
    p = argparse.ArgumentParser()
    p.add_argument("suite", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    analysis = json.loads((args.suite/"analysis.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    jobs = [(shape, "residual", True) for shape in ("cube", "sphere", "cuboid")]
    jobs += [("cube", arm, False) for arm in ("pure", "bc", "residual")]
    for shape, arm, zero in jobs:
        record = next(r for r in analysis["runs"] if (r["task"], r["arm"], r["seed"]) == (shape, arm, 0))
        run = Path(record["run"])
        source = json.loads((run/"run_config.json").read_text())
        verify_sources(run, source)
        name = f"prior_{shape}" if zero else f"repeat_cube_{arm}_seed0"
        output = args.output/name
        command = [sys.executable, "scripts/train_orca_residual.py", "--arm", arm,
                   "--shape", shape, "--seed", "0", "--bound", str(source["bound"]),
                   "--reference", str(source["reference"]), "--eval-start", "2000",
                   "--eval-count", "32", "--output", str(output), "--steps", "0", "--confirmatory"]
        if zero:
            command.append("--zero")
        else:
            command += ["--checkpoint", str(run/f"step_{record['best_step']:09d}/params.pkl")]
        print("EVALUATE", name, flush=True)
        subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)
        meta = json.loads((output/"run_config.json").read_text())
        assert meta["model_sha256"] == source["model_sha256"]
        assert (output/"env_config.json").read_bytes() == (run/"env_config.json").read_bytes()
        results.append(dict(name=name, source_run=str(run), evaluation=str(output),
                            command=command, summary=json.loads((output/"summary.json").read_text())))
        (args.output/"results.json").write_text(json.dumps(results, indent=2))
    (args.output/"completed.json").write_text(json.dumps(dict(completed=True, runs=len(results)), indent=2))


if __name__ == "__main__":
    main()
