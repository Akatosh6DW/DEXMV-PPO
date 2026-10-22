"""Execute all 110 fixed-policy cap tests, optionally as disjoint shards."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from audit_oakink_cap_evaluation import audit
from evaluate_oakink_cap import SOURCES
from oakink_cap_protocol import ROOT, check_protocol, recipes, scenarios
from prepare_hic_orca_object import digest

EVAL_SOURCES = (*SOURCES, "audit_oakink_cap_evaluation.py", "run_oakink_cap_evaluations.py")


def jobs():
    result = [dict(run=r["label"], scenario=s, baseline=None, label=r["label"] + "__" + s)
              for r in recipes() for s in scenarios()]
    result += [dict(run="s401_residual", scenario=s, baseline=b, label="zero_" + b + "__" + s)
               for b in ("video", "initial") for s in scenarios()]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    args = p.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        p.error("Invalid shard")
    suite, output = args.suite.resolve(), args.output.resolve()
    check_protocol(suite / "protocol.json")
    for recipe in recipes():
        assert json.loads((suite / recipe["label"] / "formal_audit.json").read_text())["passed"]
    expected = jobs()[args.shard_index::args.shard_count]
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["jobs"] == expected and manifest["suite"] == str(suite)
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = dict(suite=str(suite), jobs=expected, protocol_sha256=digest(suite / "protocol.json"),
                        sources={n: digest(ROOT / "scripts" / n) for n in EVAL_SOURCES})
        (output / "sources").mkdir()
        for name in EVAL_SOURCES:
            shutil.copyfile(ROOT / "scripts" / name, output / "sources" / name)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = []
        for job in expected:
            check_protocol(suite / "protocol.json")
            assert digest(suite / "protocol.json") == manifest["protocol_sha256"]
            for name, expected_hash in manifest["sources"].items():
                assert digest(ROOT / "scripts" / name) == expected_hash, "Frozen cap evaluator changed"
            directory = output / job["label"]
            if not (directory / "report.json").exists():
                if directory.exists():
                    raise RuntimeError("Partial cap evaluation retained: " + str(directory))
                if shutil.disk_usage(output).free < 5 * 1024**3:
                    raise RuntimeError("Less than 5 GiB free")
                command = [sys.executable, "scripts/evaluate_oakink_cap.py", "--run", str(suite / job["run"]),
                           "--scenario", job["scenario"], "--output", str(directory)]
                if job["baseline"]:
                    command += ["--baseline", job["baseline"]]
                with (output / (job["label"] + ".log")).open("x") as log:
                    env = dict(os.environ, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    (output / "live.json").write_text(json.dumps(dict(label=job["label"], state="running", pid=process.pid)))
                    print("EVALUATE", job["label"], process.pid, flush=True)
                    code = process.wait()
                (output / "live.json").write_text(json.dumps(dict(label=job["label"], state="exited", code=code)))
                if code:
                    raise RuntimeError("Cap evaluation failed: " + job["label"])
            checked = audit(directory, job, suite)
            (directory / "independent_audit.json").write_text(json.dumps(checked, indent=2) + "\n")
            report = json.loads((directory / "report.json").read_text())
            completed.append(dict(label=job["label"], **report["metrics"]))
            (output / "progress.json").write_text(json.dumps(completed, indent=2) + "\n")
            print("AUDITED", job["label"], report["metrics"]["success"], flush=True)
        (output / "complete.json").write_text(json.dumps(dict(jobs=len(completed), manifest_sha256=digest(output / "manifest.json"))))


if __name__ == "__main__":
    main()
