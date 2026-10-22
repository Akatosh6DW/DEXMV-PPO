"""Run the fixed cap comparison, preserving failed policies and audit failures."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from audit_oakink_cap_run import audit_run
from oakink_cap_protocol import ROOT, check_protocol, freeze
from prepare_hic_orca_object import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gate", type=Path)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    output = args.output.resolve()
    if not args.resume:
        if args.gate is None:
            p.error("New studies require a GPU gate")
        output.mkdir(parents=True, exist_ok=False)
        freeze(output, args.gate)
    with (output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol_path = output / "protocol.json"
        protocol = check_protocol(protocol_path)
        completed = []
        for recipe in protocol["recipes"]:
            check_protocol(protocol_path)
            run = output / recipe["label"]
            if not (run / "complete.json").exists():
                if run.exists():
                    raise RuntimeError("Partial cap run retained for inspection: " + str(run))
                if shutil.disk_usage(output).free < 5 * 1024**3:
                    raise RuntimeError("Less than 5 GiB free")
                command = [sys.executable, "scripts/train_oakink_cap.py", "--output", str(run), "--protocol", str(protocol_path),
                           "--method", recipe["method"], "--seed", str(recipe["seed"]), "--steps", str(recipe["steps"])]
                for key in ("model", "reference", "gate"):
                    command += ["--" + key, protocol[key]]
                for key, value in protocol["configuration"].items():
                    command += ["--" + key.replace("_", "-"), str(value)]
                if recipe["method"] == "bc":
                    command += ["--dataset", protocol["dataset"]]
                elif recipe["method"] == "guided":
                    command += ["--bc-checkpoint", str(output / f"s{recipe['seed']}_bc/bc_params.pkl")]
                with (output / (recipe["label"] + ".log")).open("x") as log:
                    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                    env.pop("JAX_PLATFORMS", None)
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    (output / "live.json").write_text(json.dumps(dict(label=recipe["label"], pid=process.pid, state="running")))
                    print("TRAIN", recipe["label"], process.pid, flush=True)
                    code = process.wait()
                (output / "live.json").write_text(json.dumps(dict(label=recipe["label"], state="exited", exit_code=code)))
                if code:
                    raise RuntimeError("Cap training failed: " + recipe["label"])
            result = audit_run(run)
            (run / "formal_audit.json").write_text(json.dumps(result, indent=2) + "\n")
            complete = json.loads((run / "complete.json").read_text())
            completed.append(dict(label=recipe["label"], first=complete["first"], last=complete["last"], best=complete["best"]))
            (output / "progress.json").write_text(json.dumps(completed, indent=2) + "\n")
            print("AUDITED", recipe["label"], json.dumps(complete["last"]), flush=True)
        (output / "complete.json").write_text(json.dumps(dict(training_jobs=len(completed),
                protocol_sha256=digest(protocol_path), independent_evaluation_complete=False), indent=2) + "\n")


if __name__ == "__main__":
    main()
