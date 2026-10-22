"""Run the frozen multi-seed study sequentially, with audited resume and logs."""

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from audit_orca_formal_run import audit_run
from orca_formal_protocol import DATASET, GATES, MODEL, PHYSICS_GATE, REFERENCE, SEEDS, SOURCE_NAMES, STEPS, recipes, scenarios
from prepare_hic_orca_object import digest
from run_orca_matched_pilot import summarize

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def freeze(output):
    inputs = [MODEL, REFERENCE, PHYSICS_GATE, DATASET, *GATES.values()]
    inputs += [str(Path(DATASET).parent / "dataset_manifest.json")]
    framework = ROOT / "Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src"
    inputs += [str(p) for p in (framework / "manipulation/orca_hand").glob("*.py")]
    inputs += [str(framework / p) for p in ("mjx_env.py", "wrapper.py")]
    physical = json.loads((ROOT / PHYSICS_GATE).read_text())
    for path, expected in physical["asset_sha256"].items():
        assert digest(path) == expected
        inputs.append(path)
    for path in inputs:
        if not (ROOT / path).is_file():
            raise FileNotFoundError("Finish the validation gates before freezing: " + path)
    manifest = json.loads((ROOT / DATASET).with_name("dataset_manifest.json").read_text())
    assert manifest["dataset_sha256"] == digest(ROOT / DATASET)
    assert manifest["source_frames"] == [30, 95] and manifest["source_video_count"] == 1
    assert manifest["physics_backend"] == "native_mujoco_callback"
    assert manifest["model_sha256"] == digest(ROOT / MODEL) and manifest["reference_sha256"] == digest(ROOT / REFERENCE)
    source_hashes = {name: digest(ROOT / "scripts" / name) for name in SOURCE_NAMES}
    for (mode, bound), path in GATES.items():
        gate = json.loads((ROOT / path).read_text())
        assert gate["pass_paired_tactile"] and gate["reference_mode"] == mode and gate["residual_bound"] == bound
        assert gate["matched_source_sha256"] == source_hashes["orca_matched_env.py"]
        assert gate["native_source_sha256"] == source_hashes["orca_native_physics.py"]
    protocol = dict(version=1, frozen_utc=datetime.now(timezone.utc).isoformat(), seeds=SEEDS,
                    ppo_steps=STEPS, evals=9, development_seeds=list(range(11000, 11016)),
                    test_seeds=list(range(25000, 25064)), scenarios=scenarios(), recipes=recipes(),
                    source_video_count=1, dataset_sha256=digest(ROOT / DATASET),
                    inputs={str((ROOT / p).resolve()): digest(ROOT / p) for p in inputs},
                    sources=source_hashes, xla_flags=os.environ.get("XLA_FLAGS", ""),
                    packages={p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "mujoco", "brax", "flax", "optax", "numpy")},
                    selection="fixed final checkpoint", statistical_unit="independent training seed",
                    primary_scope="8 conditions x 5 seeds; shared configuration, not separately optimized algorithms",
                    complete_goal_requires="main, reference and exploration ablations; zero correction; robustness; audits/videos; matched cap task")
    output.mkdir(parents=True, exist_ok=False)
    (output / "frozen_sources").mkdir()
    for name in SOURCE_NAMES:
        shutil.copyfile(ROOT / "scripts" / name, output / "frozen_sources" / name)
    shutil.copyfile(ROOT / "docs/orca_formal_experiment.md", output / "protocol.md")
    write_json(output / "protocol.json", protocol)
    return protocol


def verify_frozen(protocol):
    for package, version in protocol["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError("Frozen package changed: " + package)
    for path, expected in protocol["inputs"].items():
        if digest(path) != expected:
            raise ValueError("Frozen input changed: " + path)
    for name, expected in protocol["sources"].items():
        if digest(ROOT / "scripts" / name) != expected:
            raise ValueError("Frozen implementation changed: " + name)
    if protocol["xla_flags"] != os.environ.get("XLA_FLAGS", ""):
        raise ValueError("Frozen runtime flags changed")


def command(recipe, output):
    args = [sys.executable, "scripts/train_orca_matched.py", "--native-physics", "--model", MODEL,
            "--reference", REFERENCE, "--gate", PHYSICS_GATE,
            "--tactile-gate", GATES[(recipe["reference_mode"], recipe["bound"])],
            "--output", str(output / recipe["label"]), "--method", recipe["method"],
            "--tactile", recipe["tactile"], "--seed", str(recipe["seed"]),
            "--steps", str(recipe["steps"]), "--evals", "9", "--eval-start", "11000", "--eval-count", "16",
            "--learning-rate", ".0003", "--std", str(recipe["std"]),
            "--residual-bound", str(recipe["bound"]), "--reference-mode", recipe["reference_mode"]]
    if recipe["method"] == "bc":
        args += ["--dataset", DATASET, "--bc-updates", "2000"]
    if recipe["bc"]:
        args += ["--bc-checkpoint", str(output / recipe["bc"] / "bc_params.pkl")]
    return args


def update_progress(output, records):
    write_json(output / "progress.json", dict(training_complete=len(records) == 80,
                                               entire_goal_complete=False, completed=len(records), records=records))
    lines = ["# Formal Study Progress", "", "Training progress only; not the final independent test report.", "",
             "| Run | Phase | First / Last / Max success | Best step | Final mean hold s | Peak force N | Max penetration mm |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in records:
        lines.append(f"| {r['run']} | {r['phase']} | {r['first']:.1%} / {r['last']:.1%} / {r['max']:.1%} | "
                     f"{r['best_step']} | {r['final_hold']:.3f} | {r['max_force_n']:.2f} | {r['penetration_max_mm']:.3f} |")
    lines += ["", f"Completed: {len(records)}/80 training jobs (40 main, 20 reference, 20 exploration including BC fits).",
              "Independent tests, uncertainty estimates, videos and the matched cap study are separate completion requirements."]
    (output / "progress.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--phase", choices=("main", "reference", "exploration", "all"), default="main")
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = json.loads((output / "protocol.json").read_text()) if args.resume else freeze(output)
    with (output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_frozen(protocol)
        records = []
        for recipe in protocol["recipes"]:
            directory = output / recipe["label"]
            if (directory / "summary.json").exists():
                audit_run(directory, recipe, protocol)
                saved = json.loads((directory / "formal_result.json").read_text())
                records.append(saved)
        update_progress(output, records)
        finished = {r["run"] for r in records}
        for recipe in protocol["recipes"]:
            if recipe["label"] in finished or (args.phase != "all" and recipe["phase"] != args.phase):
                continue
            verify_frozen(protocol)
            directory = output / recipe["label"]
            if directory.exists():
                raise RuntimeError("Partial run preserved; inspect its exit and do not restart silently: " + str(directory))
            if shutil.disk_usage(output).free < 5 * 1024**3:
                raise RuntimeError("Less than 5 GiB free; do not risk incomplete artifacts")
            cmd = command(recipe, output)
            label = recipe["label"]
            started = time.monotonic()
            with (output / (label + ".log")).open("x") as log:
                process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                write_json(output / "live.json", dict(pid=process.pid, parent_pid=os.getpid(),
                                                      label=label, command=cmd, state="running"))
                print("START", label, "pid", process.pid, flush=True)
                code = process.wait()
            write_json(output / "live.json", dict(label=label, exit_code=code, state="exited"))
            if code:
                raise RuntimeError(f"{label} exited {code}; preserved log and partial run")
            audit = audit_run(directory, recipe, protocol)
            write_json(directory / "formal_audit.json", audit)
            result = summarize(directory)
            result.update(phase=recipe["phase"], reference_mode=recipe["reference_mode"],
                          std=recipe["std"], bound=recipe["bound"], process_elapsed_sec=time.monotonic() - started,
                          next_action="Next predeclared independent seed/condition; no outcome-based retuning")
            write_json(directory / "formal_result.json", result)
            records.append(result)
            update_progress(output, records)
            print("RESULT", json.dumps(result), flush=True)
            if result["penetration_max_mm"] > 5 or result["forbidden_fraction"] > .01:
                raise RuntimeError("Contact quality gate requires inspection before the next run")
        print("PHASE FINISHED", args.phase, "completed jobs", len(records), flush=True)


if __name__ == "__main__":
    main()
