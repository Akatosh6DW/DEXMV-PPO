"""Reproducible S07 training and evaluation, with a pre-test selection freeze."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from evaluate_s06_selected import select
from prepare_cylinder_s02_source import digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/orca_benchmark_suite/S07"
S06 = OUT.parent / "S06"
DOC = ROOT / "docs/experiments/S07"
METHODS = ("ppo", "guided", "residual")
SEEDS = (7101, 7102, 7103)
BENCHMARKS = ("B0_nominal", "B1_position", "B2_yaw", "B3_friction", "B4_mass", "B5_push")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "inputs"
    if target.exists():
        manifest = json.loads((OUT / "inputs_manifest.json").read_text())
        for path, checksum in manifest.items():
            if digest(ROOT / path) != checksum:
                raise ValueError("Frozen input changed: " + path)
        return
    target.mkdir()
    for name in ("model.xml", "reference.npz"):
        shutil.copyfile(S06 / "P09_implicit" / name, target / name)
    for variant in ("linear", "uncorrected"):
        directory = OUT / ("inputs_" + variant)
        directory.mkdir()
        shutil.copyfile(target / "model.xml", directory / "model.xml")
        source = target if variant == "linear" else S06 / "P02_aligned_wrist"
        with np.load(source / "reference.npz") as z:
            packet = dict(z)
        if variant == "linear":
            phase = packet["time"] / packet["time"][-1]
            packet["ctrl"] = packet["initial_ctrl"] + phase[:, None] * (packet["ctrl"][-1]-packet["initial_ctrl"])
        with np.load(target / "reference.npz") as z:
            for key in ("initial_ctrl", "object_position", "object_quaternion"):
                np.testing.assert_array_equal(packet[key], z[key])
        np.savez_compressed(directory / "reference.npz", **packet)
    files = [*OUT.glob("inputs*/*.xml"), *OUT.glob("inputs*/*.npz"), DOC / "实验约定.md",
             S06 / "D03_implicit_demonstrations/trajectories.npz"]
    write_json(OUT / "inputs_manifest.json", {str(p.relative_to(ROOT)): digest(p) for p in files})
    write_json(OUT / "protocol.json", dict(training_seeds=SEEDS, methods=METHODS, steps=163840,
        evals=9, dev_seeds=list(range(70000, 70008)), test_seeds=list(range(74000, 74064)),
        learning_rate=1e-4, bc_updates=20000, residual_bound=.08,
        benchmarks=list(BENCHMARKS), input_manifest_sha256=digest(OUT / "inputs_manifest.json")))


def launch(directory, arguments):
    if (directory / "summary.json").exists():
        summary = json.loads((directory / "summary.json").read_text())
        if summary.get("status") != "completed":
            raise ValueError("Not completed: " + str(directory))
        return
    if directory.exists():
        raise ValueError("Partial run needs diagnosis; refusing to overwrite " + str(directory))
    directory.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "scripts/run_s07_job.py"), "--output", str(directory),
           "--height-weight", "5", *map(str, arguments)]
    log = directory.parent / (directory.name + ".log")
    print("START", directory.relative_to(ROOT), flush=True)
    with log.open("w") as f:
        f.write(json.dumps(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    print("DONE", directory.relative_to(ROOT), flush=True)


def record_training(directory):
    summary = json.loads((directory / "summary.json").read_text())
    with (directory / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    best = select(directory)
    note = dict(run=str(directory.relative_to(ROOT)), success30=summary["success30"],
                hold=summary["pad_hold_sec"], selected=best,
                max_penetration=max(float(r["max_penetration_m"]) for r in rows),
                max_forbidden_rate=max(float(r["forbidden_rate"]) for r in rows))
    write_json(directory / "iteration_audit.json", note)
    # Inspect metrics before launching the next independent replication.
    with (DOC / "iteration_log.md").open("a") as f:
        f.write(f"\n## {directory.name}\n\n")
        f.write("固定场景/奖励/超参数；只改变已约定的方法或随机种子。无续训，均重新初始化。\n\n")
        f.write(f"- 成功率 first/last/max/best_step：`{summary['success30']}`\n")
        f.write(f"- 保持时间：`{summary['pad_hold_sec']}`\n")
        f.write(f"- 最大穿透 {note['max_penetration']*1000:.3f} mm；最高禁止接触回合比例 {note['max_forbidden_rate']:.3f}。\n")
        f.write(f"- 按开发标准选中 {best['development']['step']:.0f} 步；具体命令见同名 `.log` 第一行。\n")
        f.write("- 这是复验而不是测试集调参；退化保留。下一轮执行协议中的独立种子/方法，训练完成后再冻结测试权重。\n")
    print("METRICS", json.dumps(note), flush=True)


def train_all():
    for method in METHODS:
        for seed in SEEDS:
            directory = OUT / "train" / f"{method}_s{seed}"
            args = ["--input", OUT / "inputs", "--method", method, "--seed", seed,
                    "--steps", 163840, "--evals", 9, "--eval-count", 8, "--eval-start", 70000,
                    "--learning-rate", 1e-4, "--std", .1]
            if method == "guided":
                args += ["--dataset", S06 / "D03_implicit_demonstrations/trajectories.npz", "--bc-updates", 20000]
            launch(directory, args)
            if not (directory / "iteration_audit.json").exists():
                record_training(directory)


def freeze():
    path = OUT / "selection.json"
    if path.exists():
        packet = json.loads(path.read_text())
        for item in packet["selected"].values():
            if digest(item["checkpoint"]) != item["sha256"]:
                raise ValueError("Selected checkpoint modified")
        return packet
    selected = {}
    for method in METHODS:
        for seed in SEEDS:
            name = f"{method}_s{seed}"
            item = select(OUT / "train" / name)
            item.update(method=method, train_seed=seed)
            item["bc_updates"] = 20000 if method == "guided" else 0
            selected[name] = item
            if method == "guided":
                checkpoint = OUT / "train" / name / "step_000000000/params.pkl"
                selected[f"bc_s{seed}"] = dict(method="guided", train_seed=seed, bc_updates=20000,
                    checkpoint=str(checkpoint), sha256=digest(checkpoint), training_budget=0, std=.1)
    packet = dict(selected=selected, timestamp=time.time(), protocol_sha256=digest(OUT / "protocol.json"),
                  rule="positive development step: success30, mean hold, latest; frozen before independent tests")
    write_json(path, packet)
    return packet


def evaluate_all(workers=1):
    selection = freeze()
    checksum = digest(OUT / "selection.json")
    jobs = []
    for bench in BENCHMARKS:
        count = 64 if bench == "B0_nominal" else 32
        for name, item in {"reference": None, **selection["selected"]}.items():
            jobs.append((bench, name, item, OUT / "inputs", count, 1.))
    for name, item in selection["selected"].items():
        if item["method"] == "residual":
            for gain in (0., .5):
                jobs.append(("A_gain", name+f"_gain{gain:g}", item, OUT / "inputs", 64, gain))
    for variant in ("linear", "uncorrected"):
        jobs.append(("A_reference", variant, None, OUT / ("inputs_"+variant), 64, 1.))
    def evaluate_job(job):
        bench, name, item, inputs, count, gain = job
        directory = OUT / "eval" / bench / name
        args = ["--input", inputs, "--method", item["method"] if item else "reference", "--steps", 0,
                "--eval-count", count, "--eval-start", 74000, "--benchmark", bench if bench in BENCHMARKS else "B0_nominal",
                "--seed", item["train_seed"] if item else 7101, "--residual-gain", gain]
        if bench == "B0_nominal" or bench.startswith("A_"):
            args += ["--save-all-trajectories"]
        if item:
            args += ["--checkpoint", item["checkpoint"], "--bc-updates", item["bc_updates"]]
        launch(directory, args)
        if digest(OUT / "selection.json") != checksum:
            raise ValueError("Selection changed during testing")
        episodes = json.loads((directory / "step_000000000/episodes.json").read_text())
        print("TEST", bench, name, sum(e["success30"] for e in episodes), "/", len(episodes), flush=True)
    # Workers own separate processes and output directories. Selection and
    # inputs remain immutable; model RNGs are local to each subprocess.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(evaluate_job, jobs))
    write_json(OUT / "evaluation_complete.json", dict(jobs=len(jobs), selection_sha256=checksum))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "train", "evaluate"))
    p.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    args = p.parse_args()
    prepare()
    if args.phase == "train":
        train_all()
        freeze()
    elif args.phase == "evaluate":
        evaluate_all(args.workers)


if __name__ == "__main__":
    main()
