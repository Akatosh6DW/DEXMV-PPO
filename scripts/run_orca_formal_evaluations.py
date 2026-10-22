"""Run every predeclared held-out condition, with immutable evaluator sources."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from orca_formal_protocol import recipes, scenarios
from prepare_hic_orca_object import digest
from audit_orca_formal_evaluation import audit

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("evaluate_orca_formal.py", "evaluate_orca_matched_native.py", "orca_formal_protocol.py",
           "orca_matched_env.py", "orca_residual_metrics.py", "orca_guidance_recheck.py",
           "diagnose_orca_grasp.py", "audit_orca_trace_geometry.py", "run_orca_formal_evaluations.py",
           "audit_orca_formal_evaluation.py")


def jobs(phase):
    result = []
    for recipe in recipes():
        if phase != "all" and recipe["phase"] != phase:
            continue
        if recipe["phase"] == "exploration" and recipe["method"] == "bc":
            continue
        for scenario in (scenarios() if recipe["phase"] == "main" else ["nominal"]):
            result.append(dict(run=recipe["label"], step=recipe["steps"], scenario=scenario,
                               zero=None, label=recipe["label"] + "__" + scenario))
    if phase in ("reference", "all"):
        for mode in ("video", "endpoint", "initial"):
            result.append(dict(run="s301_residual_on", step=655360, scenario="nominal", zero=mode,
                               label="zero_" + mode))
    return result


def shard_jobs(phase, index=0, count=1):
    if count < 1 or not 0 <= index < count:
        raise ValueError("Invalid evaluation shard")
    return jobs(phase)[index::count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("main", "reference", "exploration", "all"), default="main")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    suite, output = args.suite.resolve(), args.output.resolve()
    expected = shard_jobs(args.phase, args.shard_index, args.shard_count)
    for job in expected:
        if not (suite / job["run"] / "formal_audit.json").is_file():
            raise ValueError("Finish and audit all selected training jobs before held-out evaluation")
        if not json.loads((suite / job["run"] / "formal_audit.json").read_text())["passed"]:
            raise ValueError("Training audit did not pass")
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["jobs"] == expected and manifest["suite"] == str(suite)
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = dict(suite=str(suite), phase=args.phase, jobs=expected,
                        shard_index=args.shard_index, shard_count=args.shard_count,
                        protocol_sha256=digest(suite / "protocol.json"),
                        amendment_sha256=digest(ROOT / "docs/orca_formal_evaluation_amendment.md"),
                        sources={n: digest(ROOT / "scripts" / n) for n in SOURCES})
        (output / "sources").mkdir()
        for name in SOURCES:
            shutil.copyfile(ROOT / "scripts" / name, output / "sources" / name)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for job in expected:
            for name, value in manifest["sources"].items():
                assert digest(ROOT / "scripts" / name) == value, "Evaluator changed during formal testing"
            assert digest(suite / "protocol.json") == manifest["protocol_sha256"]
            assert digest(ROOT / "docs/orca_formal_evaluation_amendment.md") == manifest["amendment_sha256"]
            destination = output / job["label"]
            report_path = destination / "report.json"
            if report_path.exists():
                result = audit(destination, job, suite)
                (destination / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
                continue
            if destination.exists():
                raise RuntimeError("Partial evaluation preserved for inspection: " + str(destination))
            if shutil.disk_usage(output).free < 5 * 1024**3:
                raise RuntimeError("Less than 5 GiB free")
            command = [sys.executable, "scripts/evaluate_orca_formal.py", "--run", str(suite / job["run"]),
                       "--step", str(job["step"]), "--scenario", job["scenario"], "--output", str(destination)]
            if job["zero"]:
                command += ["--zero-reference", job["zero"]]
            env = dict(os.environ, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1")
            print("EVALUATE", job["label"], flush=True)
            with (output / (job["label"] + ".log")).open("x") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                (output / "live.json").write_text(json.dumps(dict(pid=process.pid, label=job["label"], state="running")))
                code = process.wait()
            (output / "live.json").write_text(json.dumps(dict(label=job["label"], state="exited", exit_code=code)))
            if code:
                raise RuntimeError(f"Evaluation failed: {job['label']} ({code})")
            report = json.loads(report_path.read_text())
            result = audit(destination, job, suite)
            (destination / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
            print("RESULT", job["label"], json.dumps(report["metrics"]), flush=True)
        (output / "complete.json").write_text(json.dumps(dict(jobs=len(expected), manifest_sha256=digest(output / "manifest.json"))))
        print("EVALUATION PHASE COMPLETE", args.phase, len(expected), flush=True)


if __name__ == "__main__":
    main()
