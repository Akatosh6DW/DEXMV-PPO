"""Sequential, source-matched eight-condition feasibility study (not a benchmark)."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from audit_orca_trace_geometry import audit
from diagnose_orca_refit import diagnose
from prepare_hic_orca_object import digest

ROOT = Path(__file__).resolve().parents[1]


def summarize(directory):
    with (directory / "metrics.csv").open() as stream:
        curve = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    best = max(curve, key=lambda row: row["success30"])
    last = curve[-1]
    trace = directory / f"step_{int(last['step']):09d}" / "trajectories.npz"
    diagnosis = diagnose(trace)
    (trace.parent / "diagnostics.json").write_text(json.dumps(diagnosis, indent=2) + "\n")
    config = json.loads((directory / "run_config.json").read_text())
    geometry = audit(ROOT / config["model"], trace)
    (trace.parent / "geometry_audit.json").write_text(json.dumps(geometry, indent=2) + "\n")
    return dict(run=directory.name, method=config["method"], tactile=config["tactile"],
                seed=config["seed"], first=curve[0]["success30"], last=last["success30"],
                max=best["success30"], best_step=int(best["step"]),
                initial_hold=curve[0]["mean_hold"], final_hold=last["mean_hold"],
                max_mean_hold=max(row["mean_hold"] for row in curve),
                success15=last["success15"], success15_any=last["success15_any"],
                retained=last["retained"], final_steps=int(last["step"]),
                residual_max_rad=last["requested_residual_max_rad"],
                max_force_n=last["active_force_max_n"], elapsed_sec=last["elapsed_sec"],
                bc_elapsed_sec=config["bc_elapsed_sec"],
                forbidden_fraction=diagnosis["released_forbidden_fraction"],
                penetration_max_mm=geometry["max_penetration_m"] * 1000,
                checkpoint=str(trace.parent / "params.pkl"))


def report(root, records):
    (root / "results.json").write_text(json.dumps(dict(complete=len(records) == 8, records=records), indent=2) + "\n")
    lines = ["# HIC53 Eight-Condition Pilot", "",
             "Single training seed; feasibility only, not a statistical ranking. Native MuJoCo physics with Brax PPO.",
             "One source video; generated simulator episodes are not additional videos. Teacher may drop the object.",
             "BC and guided PPO use paired BC checkpoints. Pure PPO has no online reference, but shares a video-derived reset.",
             "15s-any means ever held for 15 seconds; 15s-final and 30s-final also require a valid final frame.",
             "No live RGB input. Residual means video target + bounded learned correction at every control step.", "",
             "| Condition | PPO steps | First / Last / Max 30s-final | Best step | Mean hold first -> last (s) | 15s-any / 15s-final | Retained | Max penetration (mm) |", 
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in records:
        lines.append(f"| {r['method']} tactile={r['tactile']} | {r['final_steps']} | "
                     f"{r['first']:.1%} / {r['last']:.1%} / {r['max']:.1%} | {r['best_step']} | "
                     f"{r['initial_hold']:.3f} -> {r['final_hold']:.3f} | "
                     f"{r['success15_any']:.1%} / {r['success15']:.1%} | {r['retained']:.1%} | {r['penetration_max_mm']:.3f} |")
    lines += ["", f"Completed conditions: {len(records)}/8.", "",
              "Configuration, commands, dataset provenance, time, force and dirty-contact diagnostics are stored alongside this report.",
              "Mass/friction are assumptions; hand self-collisions remain disabled. Not hardware validation."]
    (root / "results.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=201)
    p.add_argument("--steps", type=int, default=327680)
    args = p.parse_args()
    if args.steps <= 0 or args.steps % 40960:
        p.error("Budget must be a positive multiple of 40960")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    model = "artifacts/orca_collision_refit/O20_hic53_elliptic/cube.xml"
    reference = "artifacts/orca_collision_refit/O20_hic53_elliptic/reference.npz"
    common = [sys.executable, "scripts/train_orca_matched.py", "--native-physics", "--model", model,
              "--reference", reference, "--gate", "artifacts/orca_collision_refit/O21_hic53_elliptic_validation/report.json",
              "--tactile-gate", "artifacts/orca_collision_refit/O33b_hic53_native_callback/report.json",
              "--seed", str(args.seed), "--learning-rate", ".0003"]
    protocol = dict(seed=args.seed, ppo_steps=args.steps, model_sha256=digest(ROOT / model),
                    reference_sha256=digest(ROOT / reference), physics_backend="native_mujoco_callback",
                    evaluation_seeds=list(range(10000, 10016)), teacher_seeds=list(range(8100, 8132)),
                    bc_updates=2000, std=dict(bc=.03, guided=.03, ppo=.7, residual=.7),
                    selection="fixed final checkpoint; training curves also reported",
                    scope="single-seed feasibility pilot; not multi-seed final comparison",
                    xla_flags=os.environ.get("XLA_FLAGS", ""), commands=[])
    def launch(command, label):
        protocol["commands"].append(command)
        (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        print("START", label, flush=True)
        started = time.monotonic()
        with (args.output / (label + ".log")).open("w") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f"{label} failed with code {process.returncode}; inspect its log")
        print("FINISHED", label, "seconds", time.monotonic() - started, flush=True)
    dataset = args.output / "teacher"
    launch(common + ["--mode", "collect", "--method", "ppo", "--tactile", "on",
                     "--output", str(dataset), "--eval-start", "8100", "--eval-count", "32"], "teacher")
    records = []
    report(args.output, records)
    for method in ("bc", "ppo", "guided", "residual"):
        for tactile in ("off", "on"):
            label = f"{method}_{tactile}"
            directory = args.output / label
            command = common + ["--method", method, "--tactile", tactile, "--output", str(directory),
                                "--eval-start", "10000", "--eval-count", "16",
                                "--steps", str(0 if method == "bc" else args.steps), "--evals", "5"]
            if method == "bc":
                command += ["--dataset", str(dataset / "trajectories.npz")]
            elif method == "guided":
                command += ["--bc-checkpoint", str(args.output / f"bc_{tactile}" / "bc_params.pkl")]
            launch(command, label)
            record = summarize(directory)
            records.append(record)
            report(args.output, records)
            print("RESULT", json.dumps(record), flush=True)
            if record["penetration_max_mm"] > 5 or record["forbidden_fraction"] > .01:
                raise RuntimeError("Dirty contact in pilot; inspect before proceeding")


if __name__ == "__main__":
    main()
