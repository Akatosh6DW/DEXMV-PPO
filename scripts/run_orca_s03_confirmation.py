"""Freeze and execute the S03 confirmatory matrix after bounded development."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from run_orca_s03_development import BASE, frozen_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/orca_benchmark_suite/S03"
SEEDS = list(range(21001, 21011))
BENCHMARKS = ["B0_nominal", "B1_position", "B2_yaw", "B3_friction", "B4_mass", "B5_push"]
BUDGETS = [163840, 655360]
GROUPS = {"ppo_fixed": "ppo", "ppo_tuned": "ppo", "guided_tuned": "guided", "residual_tuned": "residual"}
FORCES = [.15, .3, .6, 1.2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recipe_for(group, selected):
    return BASE["ppo"] if group == "ppo_fixed" else selected[GROUPS[group]]


def training_jobs(selected):
    for group, method in GROUPS.items():
        for seed in SEEDS:
            yield dict(output=f"artifacts/orca_benchmark_suite/S03/confirmation/{group}/s{seed}",
                       group=group, method=method, seed=seed, stage="confirmation", steps=655360,
                       evals=9, eval_start=61000, eval_count=16, **recipe_for(group, selected))


def evaluation_jobs(selected, force):
    conditions = [(b, None, 71000) for b in BENCHMARKS]
    conditions += [("X_"+shape, shape, 73000) for shape in ("cuboid", "sphere")]
    for condition, shape, start in conditions:
        task = dict(shape=shape) if shape else dict(benchmark=condition)
        for group, method in GROUPS.items():
            for budget in BUDGETS:
                for seed in SEEDS:
                    yield dict(output=f"artifacts/orca_benchmark_suite/S03/eval/{condition}/{group}/{budget}/s{seed}",
                               group=group, method=method, seed=seed, budget=budget, condition=condition,
                               stage="evaluation", eval_start=start, eval_count=200,
                               checkpoint=f"artifacts/orca_benchmark_suite/S03/confirmation/{group}/s{seed}/step_{budget:09d}/params.pkl",
                               **recipe_for(group, selected), **task)
        for seed in SEEDS:
            yield dict(output=f"artifacts/orca_benchmark_suite/S03/eval/{condition}/bc/s{seed}",
                       group="bc", method="bc", seed=seed, budget=0, condition=condition,
                       stage="evaluation", eval_start=start, eval_count=200,
                       checkpoint=f"artifacts/orca_benchmark_suite/S03/confirmation/guided_tuned/s{seed}/bc_params.pkl",
                       **selected["guided"], **task)
        yield dict(output=f"artifacts/orca_benchmark_suite/S03/eval/{condition}/zero",
                   group="zero", method="zero", seed=21001, budget=0, condition=condition,
                   stage="evaluation", eval_start=start, eval_count=200, std=.3, shaping=1., **task)
    for seed in SEEDS:
        yield dict(output=f"artifacts/orca_benchmark_suite/S03/mechanism/s{seed}",
                   group="residual_tuned", method="residual", seed=seed, budget=655360,
                   condition="mechanism", stage="mechanism", eval_start=72000, eval_count=200,
                   paired_replay=True, push_force=force,
                   checkpoint=f"artifacts/orca_benchmark_suite/S03/confirmation/residual_tuned/s{seed}/step_000655360/params.pkl",
                   **selected["residual"])


def select_force(results):
    candidates = []
    for force in FORCES:
        episodes = results[force]
        pre = [row for row in episodes if row["valid_before_push"]]
        if not pre:
            raise ValueError("No valid reference grasps before force; cannot calibrate perturbation")
        rate = sum(row["valid_one_second_after_push"] for row in pre) / len(pre)
        candidates.append(dict(force_n=force, pre_valid=len(pre), total=len(episodes), post_valid_rate=rate))
    chosen = min(candidates, key=lambda row: (abs(row["post_valid_rate"]-.5), row["force_n"]))
    return dict(selected_force_n=chosen["force_n"], ladder=candidates,
                rule="closest reference-only conditional survival to 0.5; ties take smallest force",
                informative=any(0 < row["post_valid_rate"] < 1 for row in candidates))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    selected_path = OUT / "selected_recipes.json"
    selection = json.loads(selected_path.read_text())
    if selection["completed_trials"] != 24:
        raise ValueError("Development is incomplete")
    gates_path = OUT / "object_geometry/gates.json"
    gates = json.loads(gates_path.read_text())
    if not gates["passed"] or not gates["policy_performance_not_used"]:
        raise ValueError("Object preparation gates are incomplete")
    for name, expected in gates["source_sha256"].items():
        if digest(ROOT / "scripts" / name) != expected:
            raise ValueError("Object gate implementation changed")
    selected = selection["recipes"]
    old = json.loads((OUT / "development_protocol.json").read_text())
    hashes = dict(old["source_hashes"])
    for name in ("orca_s03_objects.py", "orca_s03_evaluation.py", "run_orca_s03_eval.py",
                 "run_orca_s03_confirmation.py", "report_orca_s03.py", "plot_orca_s03.py",
                 "render_orca_s03.py", "audit_orca_s03.py"):
        hashes["scripts/"+name] = digest(ROOT / "scripts" / name)
    hashes[str(gates_path.relative_to(ROOT))] = digest(gates_path)
    training = list(training_jobs(selected))
    plan = dict(version="S03-confirmation-v1", development_sha256=digest(OUT / "development_protocol.json"),
                selected_recipes_sha256=digest(selected_path), training_jobs=training,
                budgets=BUDGETS, training_seeds=SEEDS, benchmark_names=BENCHMARKS,
                main_test_seeds=list(range(71000, 71200)), mechanism_test_seeds=list(range(72000, 72200)),
                object_test_seeds=list(range(73000, 73200)), force_ladder=FORCES,
                force_development_seeds=list(range(62000, 62016)), source_hashes=hashes,
                training_count=40, ordinary_eval_count=728, paired_mechanism_jobs=10,
                primary_comparisons=["B0_long_residual_tuned-ppo_tuned", "B0_long_residual_tuned-guided_tuned",
                                     "mechanism_long_pushed_closed-pushed_replay"],
                primary_correction="Holm-3", primary_score="native-substep clean_success30",
                test="two-sided exact paired training-seed sign-flip; crossed seed/case bootstrap 5000 draws",
                video_cases="first training seed 21001; first initial state per condition; long-budget groups plus BC and zero; mechanism both pushed modes",
                closed_replay_source="unperturbed same-state same-policy trajectory; zero residual after source termination")
    frozen_json(OUT / "confirmation_protocol.json", plan)
    for path, expected in hashes.items():
        if digest(ROOT / path) != expected:
            raise ValueError("Frozen input changed: " + path)
    for job in training:
        name = job["output"].split("/S03/")[1].replace("/", "__")
        frozen_json(OUT / "jobs" / (name+".json"), job)
    if args.prepare_only:
        print("Prepared 40 confirmation jobs; no training or final test executed")
        return
    lock = threading.Lock()

    def record(event):
        with lock:
            event = dict(epoch_time=time.time(), **event)
            with (OUT / "confirmation_execution.jsonl").open("a") as stream:
                stream.write(json.dumps(event)+"\n")
            print(json.dumps(event), flush=True)

    def execute(job):
        name = job["output"].split("/S03/")[1].replace("/", "__")
        path = OUT / "jobs" / (name+".json")
        frozen_json(path, job)
        output = ROOT / job["output"]
        if (output / "summary.json").exists():
            result = json.loads((output / "summary.json").read_text())
            if result["status"] != "completed" or result["job_sha256"] != digest(path):
                raise ValueError("Invalid completion: " + name)
            return
        if output.exists():
            raise FileExistsError("Inspect preserved partial job: " + str(output))
        if shutil.disk_usage(ROOT).free < 1_000_000_000:
            raise RuntimeError("Less than 1GB free; preserve artifacts before adding jobs")
        runner = "run_orca_s03_job.py" if job["stage"] == "confirmation" else "run_orca_s03_eval.py"
        command = [sys.executable, str(ROOT / "scripts" / runner), "--job", str(path)]
        output.parent.mkdir(parents=True, exist_ok=True)
        record(dict(event="start", job=job["output"], command=command))
        with output.with_suffix(".log").open("x") as log:
            subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                    XLA_PYTHON_CLIENT_PREALLOCATE="false",
                                    XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops=true"))
        result = json.loads((output / "summary.json").read_text())
        record(dict(event="completed", job=job["output"], summary=result))

    def run_jobs(jobs):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(execute, job) for job in jobs]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    run_jobs(training)
    calibration = [dict(output=f"artifacts/orca_benchmark_suite/S03/force_development/f{int(f*1000)}",
                        method="zero", group="zero", seed=21001, std=.3, shaping=1.,
                        stage="force_development", eval_start=62000, eval_count=16, push_force=f) for f in FORCES]
    run_jobs(calibration)
    force_result = select_force({job["push_force"]: json.loads((ROOT / job["output"] / "evaluation/episodes.json").read_text())
                                for job in calibration})
    frozen_json(OUT / "force_selection.json", force_result)
    evaluation = list(evaluation_jobs(selected, force_result["selected_force_n"]))
    if len(evaluation) != 738:
        raise ValueError("Unexpected final matrix count")
    frozen_json(OUT / "evaluation_protocol.json", dict(jobs=evaluation,
                confirmation_protocol_sha256=digest(OUT / "confirmation_protocol.json"),
                force_selection_sha256=digest(OUT / "force_selection.json")))
    record(dict(event="final_evaluation_frozen", jobs=len(evaluation), force=force_result))
    run_jobs(evaluation)
    frozen_json(OUT / "matrix_completion.json", dict(confirmation=40, calibration=4, evaluation=738,
                ordinary_episodes=728*200, mechanism_episodes=10*4*200, final_test_episodes=153600,
                reports_and_videos_pending=True))
    record(dict(event="matrix_complete", next="Statistics, videos, failure analysis and final audit remain"))
    for name, arguments in (("report_orca_s03.py", []), ("plot_orca_s03.py", []),
                            ("render_orca_s03.py", []), ("audit_orca_s03.py", ["--require-videos"])):
        record(dict(event="delivery_start", script=name))
        with (OUT / (name.removesuffix(".py")+".log")).open("a") as log:
            subprocess.run([sys.executable, str(ROOT / "scripts" / name), *arguments], cwd=ROOT,
                           env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        record(dict(event="delivery_complete", script=name))
    frozen_json(OUT / "automated_completion.json", dict(matrix_complete=True, statistics=True,
                plots=True, videos=100, audit_sha256=digest(OUT / "audit.json"),
                human_interpretation_and_visual_review_pending=True))
    record(dict(event="automated_delivery_complete", next="Inspect results, images and limitations; write final interpretation"))


if __name__ == "__main__":
    main()
