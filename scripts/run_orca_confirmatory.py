#!/usr/bin/env python3
"""Run the frozen paired protocol serially, preserving every subprocess log."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from analyze_orca_comparison import summarize


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/orca30/confirmatory_f01"
DATASET = ROOT / "artifacts/orca30/video3d_demos_v1/trajectories.npz"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    source_paths = [ROOT / "scripts" / name for name in (
        "train_orca_comparison.py", "orca_benchmark_env.py",
        "evaluate_orca_policy.py", "analyze_orca_comparison.py",
        "run_orca_confirmatory.py")]
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in source_paths}
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    source_sha256=sources,
                    dataset_sha256=hashlib.sha256(DATASET.read_bytes()).hexdigest(),
                    protocol="docs/orca_30s_experiment.md#frozen-confirmatory-protocol-f01",
                    training_seeds=list(range(5)), curve_seeds=list(range(1000, 1016)),
                    heldout_seeds=list(range(2000, 2032)))
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for seed in range(5):
        arms = ("scratch", "bc") if seed % 2 == 0 else ("bc", "scratch")
        for arm in arms:
            for name, expected in sources.items():
                if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"Frozen source changed: {name}")
            path = OUTPUT / f"{arm}_seed{seed}"
            cmd = [sys.executable, str(ROOT / "scripts/train_orca_comparison.py"),
                   "--arm", arm, "--seed", str(seed), "--steps", "1310720",
                   "--evals", "17", "--eval-start", "1000", "--eval-count", "16",
                   "--learning-rate", "3e-5", "--init-std", ".7", "--confirmatory",
                   "--dataset", str(DATASET), "--output", str(path)]
            if arm == "bc":
                cmd += ["--bc-steps", "2000", "--demo-weight", "1000"]
            print("START", " ".join(cmd), flush=True)
            start = time.monotonic()
            with (OUTPUT / f"{arm}_seed{seed}.log").open("w") as log:
                result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Training failed: {path}; see subprocess log")
            summary = summarize(path)
            if summary["last_step"] != 1310720 or len(summary["rows"]) != 17:
                raise RuntimeError(f"Incomplete evaluation schedule: {path}")
            summary["process_wall_sec"] = time.monotonic() - start
            rows = summary["rows"]
            best = max(rows, key=lambda r: (r["success30"], r["mean_hold"], -r["step"]))
            summary["selected_step"] = int(best["step"])
            summary["mean_hold_first_last_max"] = [rows[0]["mean_hold"], rows[-1]["mean_hold"],
                                                     max(r["mean_hold"] for r in rows)]
            (path / "training_summary.json").write_text(json.dumps(summary, indent=2))
            with (OUTPUT / "completed.jsonl").open("a") as log:
                log.write(json.dumps({k: v for k, v in summary.items() if k != "rows"}) + "\n")
            print("COMPLETE", json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)
    print("ALL_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
