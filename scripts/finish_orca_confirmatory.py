#!/usr/bin/env python3
"""Analyze every frozen run and evaluate selected and terminal checkpoints."""

import json
import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/orca30/confirmatory_f01"


def main():
    scratch = [BASE / f"scratch_seed{s}" for s in range(5)]
    guided = [BASE / f"bc_seed{s}" for s in range(5)]
    frozen = json.loads((BASE / "manifest.json").read_text())
    for name, expected in frozen["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {name}")
    reference = json.loads((scratch[0] / "run_config.json").read_text())
    for path in scratch + guided:
        summary = json.loads((path / "training_summary.json").read_text())
        if [r["step"] for r in summary["rows"]] != list(range(0, 1310721, 81920)):
            raise ValueError(f"Incomplete run: {path}")
        cfg = json.loads((path / "run_config.json").read_text())
        if cfg["development"] or not cfg["confirmatory"]:
            raise ValueError(f"Development run included: {path}")
        for name in ("model_sha256", "source_sha256"):
            if cfg[name] != reference[name]:
                raise ValueError(f"Inconsistent {name}: {path}")
        if cfg["arm"] == "bc" and cfg["dataset_sha256"] != frozen["dataset_sha256"]:
            raise ValueError(f"Dataset changed: {path}")
        best = max(summary["rows"], key=lambda r: (r["success30"], r["mean_hold"], -r["step"]))
        if int(best["step"]) != summary["selected_step"]:
            raise ValueError(f"Checkpoint selection changed: {path}")
    (BASE / "software_versions.json").write_text(json.dumps(
        {name: importlib.metadata.version(name) for name in
         ("jax", "jaxlib", "brax", "mujoco", "mujoco-mjx", "numpy", "optax", "flax")}, indent=2))
    subprocess.run([sys.executable, str(ROOT / "scripts/analyze_orca_comparison.py"),
                    "--scratch", *map(str, scratch), "--guided", *map(str, guided),
                    "--output", str(BASE / "analysis")], cwd=ROOT, check=True)
    env = dict(os.environ, JAX_COMPILATION_CACHE_DIR=str(ROOT / "artifacts/orca30/jax_cache"))
    records = []
    for seed in range(5):
        for arm in ("scratch", "bc"):
            path = BASE / f"{arm}_seed{seed}"
            summary = json.loads((path / "training_summary.json").read_text())
            for kind, step in (("selected", summary["selected_step"]), ("terminal", 1310720)):
                checkpoint = path / f"step_{step:09d}"
                output = path / f"heldout_step_{step:09d}"
                report_path = output / "report.json"
                if not report_path.exists():
                    print("EVALUATE", checkpoint, flush=True)
                    with (path / f"heldout_step_{step:09d}.log").open("w") as log:
                        subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_orca_policy.py"),
                                        str(checkpoint), "--output", str(output),
                                        "--seed-start", "2000", "--count", "32"],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                report = json.loads(report_path.read_text())
                if report["seeds"] != list(range(2000, 2032)) or not report["independent_counter_matches"]:
                    raise ValueError(f"Invalid held-out report: {report_path}")
                record = dict(seed=seed, arm=arm, kind=kind, step=step, **report)
                records.append(record)
                (BASE / "heldout_summary.json").write_text(json.dumps(records, indent=2))
                print("HELDOUT", json.dumps({k: v for k, v in record.items()
                                            if k not in ("seeds", "hold_sec")}), flush=True)
    print("ALL_EVALUATIONS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
