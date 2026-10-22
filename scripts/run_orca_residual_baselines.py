"""Sequential GPU baselines with persistent commands and per-task iteration logs."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/orca_residual"


def main():
    for ident, shape in (("R02", "sphere"), ("R03", "cuboid"), ("R04", "cube")):
        output = OUT/f"{ident}_prior_{shape}"
        command = [sys.executable, "scripts/train_orca_residual.py", "--arm", "residual",
                   "--shape", shape, "--zero", "--eval-count", "8", "--output", str(output)]
        record = dict(id=ident, task=shape, command=command, checkpoint=None,
                      change="object geometry only; same cube-video prior, physics, reward, reset jitter",
                      reason="measure cross-object open-loop prior before adding PPO",
                      expected="unknown transfer; zero correction must exactly reproduce processed prior",
                      smoke="4 residual contract unit tests passed; R01 cube GPU audit passed")
        print("START", ident, shape, flush=True)
        result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=False)
        record["returncode"] = result.returncode
        if result.returncode == 0:
            rows = list(csv.DictReader((output/"metrics.csv").open()))
            record["metrics"] = rows
            record["summary"] = json.loads((output/"summary.json").read_text())
            record["actual"] = "all per-step control and continuous-contact audits passed"
            record["next"] = "paired short PPO pilot; no reward/geometry tuning per arm"
        else:
            record["actual"] = "failed; inspect console and preserved partial output before retry"
        with (OUT/f"{shape}_iterations.jsonl").open("a") as f:
            f.write(json.dumps(record)+"\n")
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
