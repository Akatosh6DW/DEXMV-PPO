"""Bounded, log-driven coordinate search with equal trial budgets for S03."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/orca_benchmark_suite/S03"
METHODS = ("ppo", "guided", "residual")
SEEDS = (20001, 20002)
BASE = {m: dict(std=.7 if m == "ppo" else .3,
                learning_rate=3e-5 if m == "guided" else 3e-4, shaping=1.) for m in METHODS}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate(method, round_index, incumbent):
    result = dict(incumbent)
    if round_index == 0:
        return dict(BASE[method])
    if round_index == 1:
        result["std"] = .3 if method == "ppo" else .7
    elif round_index == 2:
        result["learning_rate"] /= 4
    elif round_index == 3:
        result["shaping"] = 0.
    else:
        raise ValueError("Only the four preregistered rounds are allowed")
    return result


def rank(summaries):
    if len(summaries) != len(SEEDS) or any(s["status"] != "completed" for s in summaries):
        raise ValueError("Selection requires both completed development seeds")
    mean = lambda key: sum(s[key] for s in summaries) / len(summaries)
    return (mean("clean_auc"), mean("last"), -mean("final_force_n"))


def frozen_json(path, data):
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError("Existing frozen data differ: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(json.dumps(data, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = path.read_bytes().split(b"\0")
        except OSError:
            continue
        if len(command) > 1 and Path(os.fsdecode(command[1])).name == "run_orca_s03_job.py":
            raise RuntimeError("An S03 job is already live; observe it rather than restart")
    old = json.loads((ROOT / "artifacts/orca_benchmark_suite/S02/protocol.json").read_text())
    hashes = dict(old["source_hashes"])
    for path in ("orca_s03_env.py", "run_orca_s03_job.py", "run_orca_s03_development.py"):
        hashes["scripts/"+path] = digest(ROOT / "scripts" / path)
    protocol = dict(version="S03-development-v1", methods=METHODS, seeds=SEEDS,
                    eval_seeds=list(range(61000, 61016)), steps_per_trial=163840,
                    rounds=4, trial_count=24, base_recipes=BASE,
                    selection="mean clean AUC; then final clean success; then lower force; exact ties keep incumbent",
                    test_seeds_not_accessed=[71000, 72000, 73000], source_hashes=hashes)
    # JSON converts tuples to lists; compare using the serialized representation.
    protocol = json.loads(json.dumps(protocol))
    frozen_json(OUT / "development_protocol.json", protocol)
    for path, expected in protocol["source_hashes"].items():
        if digest(ROOT / path) != expected:
            raise ValueError("Frozen source changed: " + path)
    events = OUT / "development_execution.jsonl"

    def record(event):
        event = dict(epoch_time=time.time(), **event)
        with events.open("a") as stream:
            stream.write(json.dumps(event)+"\n")
        print(json.dumps(event), flush=True)

    def execute(path):
        job = json.loads(path.read_text())
        output = ROOT / job["output"]
        if (output / "summary.json").exists():
            summary = json.loads((output / "summary.json").read_text())
            if summary["status"] != "completed" or summary["job_sha256"] != digest(path):
                raise ValueError("Invalid existing job completion")
            return job, summary
        if output.exists():
            raise FileExistsError("Preserve and inspect partial attempt before retry: " + str(output))
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(ROOT / "scripts/run_orca_s03_job.py"), "--job", str(path)]
        record(dict(event="start", job=job["output"], command=command))
        with output.with_suffix(".log").open("x") as log:
            subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                    XLA_PYTHON_CLIENT_PREALLOCATE="false",
                                    XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops=true"))
        summary = json.loads((output / "summary.json").read_text())
        with (output / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        values = [float(r["clean_success30"]) for r in rows]
        assert summary["first"] == values[0] and summary["last"] == values[-1]
        assert summary["max"] == max(values) and summary["final_step"] == 163840
        assert summary["best_step"] == int(rows[values.index(max(values))]["step"])
        record(dict(event="completed", job=job["output"], summary=summary))
        return job, summary

    incumbent = {m: dict(BASE[m]) for m in METHODS}
    best_ranks = {m: None for m in METHODS}
    selected_round = {m: 0 for m in METHODS}
    ledger = []
    for round_index in range(4):
        recipes = {m: candidate(m, round_index, incumbent[m]) for m in METHODS}
        paths = []
        for method in METHODS:
            for seed in SEEDS:
                name = f"development/{method}/r{round_index}_s{seed}"
                job = dict(output=str((OUT / name).relative_to(ROOT)), method=method, seed=seed,
                           stage="development", round=round_index, steps=163840, evals=3,
                           eval_start=61000, eval_count=16, **recipes[method])
                path = OUT / "jobs" / (name.replace("/", "__") + ".json")
                frozen_json(path, job)
                paths.append(path)
        summaries = {m: [] for m in METHODS}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(execute, path) for path in paths]
            try:
                for future in as_completed(futures):
                    job, summary = future.result()
                    summaries[job["method"]].append(summary)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        decisions = {}
        for method in METHODS:
            score = rank(summaries[method])
            accepted = best_ranks[method] is None or score > best_ranks[method]
            if accepted:
                incumbent[method], best_ranks[method], selected_round[method] = recipes[method], score, round_index
            decisions[method] = dict(candidate=recipes[method], score=score, accepted=accepted,
                                     retained_recipe=incumbent[method], retained_round=selected_round[method],
                                     trial_summaries=summaries[method])
        result = json.loads(json.dumps(dict(round=round_index, decisions=decisions)))
        # Async completion order is irrelevant; make frozen decisions deterministic.
        for item in result["decisions"].values():
            item["trial_summaries"].sort(key=lambda s: s["job_sha256"])
        frozen_json(OUT / "development_rounds" / f"r{round_index}.json", result)
        ledger.append(result)
        record(dict(event="round_decision", **result))
        lines = ["# S03 开发阶段决策", "", f"已完成{round_index+1}/4轮；不包含正式测试结果。", "",
                 "| 轮次 | 方法 | 平均清洁AUC | 平均最终清洁率 | 保留候选 | 最终保留轮次 |",
                 "|---|---|---|---|---|---|"]
        for entry in ledger:
            for method, d in entry["decisions"].items():
                lines.append(f"| {entry['round']} | {method} | {d['score'][0]:.4f} | "
                             f"{d['score'][1]:.4f} | {d['accepted']} | {d['retained_round']} |")
        (OUT / "development_report.md").write_text("\n".join(lines)+"\n")
    frozen_json(OUT / "selected_recipes.json", dict(recipes=incumbent, selected_round=selected_round,
                source_protocol_sha256=digest(OUT / "development_protocol.json"),
                completed_trials=24, online_steps_per_method=8*163840,
                total_online_development_steps=24*163840, confirmation_not_started=True))
    record(dict(event="development_complete", selected=incumbent,
                next_stage="Freeze confirmatory jobs; no test results used in recipe selection"))


if __name__ == "__main__":
    main()
