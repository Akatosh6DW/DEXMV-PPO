"""Focused, prior-friendly low-interaction benchmark using fixed S03 checkpoints."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from audit_orca_s03 import verify_trace
from report_orca_suite import bootstrap_interval, holm, sign_flip_pvalue, write_csv
from run_orca_s03_confirmation import ROOT, digest
from run_orca_s03_development import frozen_json

OUT = ROOT / "artifacts/orca_benchmark_suite/S04"
OLD = ROOT / "artifacts/orca_benchmark_suite/S03"
GROUPS = {"ppo": "ppo_tuned", "guided": "guided_tuned", "residual": "residual_tuned"}
LABELS = {"ppo": "PPO", "guided": "BC + PPO", "residual": "Reference + residual PPO",
          "bc": "BC only", "zero": "Reference only"}
SEEDS = list(range(21001, 21011))
BUDGET = 163840
CONDITIONS = ("B0_nominal", "B1_position", "B2_yaw", "B3_friction", "B4_mass")


def jobs(recipes):
    for method in ("ppo", "guided", "residual", "bc", "zero"):
        for seed in ([21001] if method == "zero" else SEEDS):
            recipe = recipes["guided" if method == "bc" else "residual" if method == "zero" else method]
            job = dict(output=f"artifacts/orca_benchmark_suite/S04/eval/{method}/s{seed}",
                       method=method, group=method, seed=seed, budget=BUDGET if method in GROUPS else 0,
                       stage="evaluation", condition="prior_friendly_low_budget", benchmark="B0_nominal",
                       eval_start=81000, eval_count=100, **recipe)
            if method != "zero":
                group = "guided_tuned" if method == "bc" else GROUPS[method]
                leaf = "bc_params.pkl" if method == "bc" else f"step_{BUDGET:09d}/params.pkl"
                job["checkpoint"] = f"artifacts/orca_benchmark_suite/S03/confirmation/{group}/s{seed}/{leaf}"
            yield job


def prepare():
    old = json.loads((OLD / "confirmation_protocol.json").read_text())
    recipes = json.loads((OLD / "selected_recipes.json").read_text())["recipes"]
    plan = list(jobs(recipes))
    hashes = dict(old["source_hashes"])
    for path in (Path(__file__), ROOT / "scripts/report_orca_suite.py", ROOT / "scripts/render_orca_modes.py"):
        hashes[str(path.relative_to(ROOT))] = digest(path)
    for job in plan:
        if job.get("checkpoint"):
            hashes[job["checkpoint"]] = digest(ROOT / job["checkpoint"])
    for group in GROUPS.values():
        for seed in SEEDS:
            path = OLD / "confirmation" / group / f"s{seed}" / "metrics.csv"
            hashes[str(path.relative_to(ROOT))] = digest(path)
            for condition in CONDITIONS:
                for budget in (BUDGET, 655360):
                    path = OLD / "eval" / condition / group / str(budget) / f"s{seed}" / "evaluation/episodes.json"
                    hashes[str(path.relative_to(ROOT))] = digest(path)
    protocol = dict(version="S04-v1", budget=BUDGET, training_seeds=SEEDS,
                    fresh_test_seeds=list(range(81000, 81100)), jobs=plan, source_hashes=hashes,
                    design="Prior-friendly task and budget chosen after S03 results; new test starts, reused policies, not independent retraining",
                    primary_contrasts=["guided-ppo", "residual-ppo"], correction="Holm-2",
                    primary_metric="clean_success30", other_metrics=["success30", "full_episode_clean_success30"],
                    bc_updates=4000, prior_cost_not_zero=True,
                    videos="seed21001, case81000 for each of five methods; STL and collision",
                    prior_report_sha256=digest(OLD / "report/statistics.json"))
    frozen_json(OUT / "protocol.json", protocol)
    for job in plan:
        frozen_json(OUT / "jobs" / f"{job['method']}_s{job['seed']}.json", job)
    return protocol


def load_protocol():
    protocol = json.loads((OUT / "protocol.json").read_text())
    for path, expected in protocol["source_hashes"].items():
        if digest(ROOT / path) != expected:
            raise ValueError("Frozen input changed: " + path)
    assert len(protocol["jobs"]) == 41
    return protocol


def run(protocol):
    def execute(job):
        path = OUT / "jobs" / f"{job['method']}_s{job['seed']}.json"
        out = ROOT / job["output"]
        if (out / "summary.json").exists():
            summary = json.loads((out / "summary.json").read_text())
            assert summary["status"] == "completed" and summary["job_sha256"] == digest(path)
            return
        if out.exists():
            raise FileExistsError("Preserve interrupted attempt before retry: " + str(out))
        out.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(dict(event="start", job=job, epoch=time.time())), flush=True)
        with out.with_suffix(".log").open("x") as log:
            subprocess.run([sys.executable, str(ROOT / "scripts/run_orca_s03_eval.py"), "--job", str(path)],
                           cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                                    XLA_PYTHON_CLIENT_PREALLOCATE="false",
                                    XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops=true"))
        print(json.dumps(dict(event="completed", job=job["output"], epoch=time.time())), flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(execute, job) for job in protocol["jobs"]]
        for future in futures:
            future.result()


def collect(protocol):
    groups, per_seed, all_rows = {}, [], []
    starts = None
    for job in protocol["jobs"]:
        run = ROOT / job["output"]
        summary = json.loads((run / "summary.json").read_text())
        config = json.loads((run / "run_config.json").read_text())
        path = OUT / "jobs" / f"{job['method']}_s{job['seed']}.json"
        assert json.loads(path.read_text()) == job
        assert summary["status"] == "completed"
        assert summary["job_sha256"] == config["job_sha256"] == digest(path)
        for key, value in job.items():
            assert config[key] == value
        for key in ("model", "reference"):
            assert digest(Path(config[key])) == config[key + "_sha256"]
        if job.get("checkpoint"):
            assert digest(ROOT / job["checkpoint"]) == config["checkpoint_sha256"]
        for filename, expected in config["source_sha256"].items():
            assert digest(run / "sources" / filename) == expected
            if "scripts/" + filename in protocol["source_hashes"]:
                assert protocol["source_hashes"]["scripts/" + filename] == expected
        d = run / "evaluation"
        rows = json.loads((d / "episodes.json").read_text())
        assert [r["seed"] for r in rows] == protocol["fresh_test_seeds"]
        for row in rows:
            assert row["clean_success30"] == (row["success30"] and not row["substep_forbidden"]
                                             and row["post_release_penetration_m"] <= .003)
        for key in ("success30", "clean_success30"):
            np.testing.assert_allclose(np.mean([r[key] for r in rows]), summary["summaries"]["evaluation"][key])
        with np.load(d / "initial_states.npz") as z:
            qpos = z["qpos"].copy()
            np.testing.assert_array_equal(z["seeds"], protocol["fresh_test_seeds"])
        assert qpos.shape == (100, 24) and np.isfinite(qpos).all()
        if starts is None:
            starts = qpos
        else:
            np.testing.assert_array_equal(starts, qpos)
        with np.load(d / "preview_trajectories.npz") as z:
            verify_trace(dict(z), rows[:4], config, 0.)
        groups.setdefault(job["method"], []).append((job["seed"], rows))
        per_seed.append(dict(method=job["method"], train_seed=job["seed"],
                             clean=np.mean([r["clean_success30"] for r in rows]),
                             raw=np.mean([r["success30"] for r in rows]),
                             eval_worker_sec=summary["elapsed_sec"]))
        all_rows.extend(dict(method=job["method"], train_seed=job["seed"], **r) for r in rows)
    for method, runs in groups.items():
        assert [seed for seed, _ in runs] == ([21001] if method == "zero" else SEEDS)
    return groups, per_seed, all_rows


def comparisons(matrices):
    result = []
    for method in ("guided", "residual"):
        delta = matrices[method] - matrices["ppo"]
        low, high = bootstrap_interval(delta)
        result.append(dict(comparison=method + "-ppo", difference_pp=float(delta.mean()*100),
                           ci_low_pp=low*100, ci_high_pp=high*100,
                           p=sign_flip_pvalue(delta.mean(axis=1)), positive_seeds=int((delta.mean(axis=1)>0).sum())))
    for row, p in zip(result, holm([r["p"] for r in result])):
        row["holm_2_p"] = p
        row["supported_on_this_benchmark"] = bool(p < .05 and row["ci_low_pp"] > 0)
    return result


def historical_report(output):
    rows = []
    for condition in CONDITIONS:
        for budget in (BUDGET, 655360):
            for method, group in GROUPS.items():
                data = []
                for seed in SEEDS:
                    path = OLD / "eval" / condition / group / str(budget) / f"s{seed}" / "evaluation/episodes.json"
                    episodes = json.loads(path.read_text())
                    assert [r["seed"] for r in episodes] == list(range(71000, 71200))
                    data.extend(episodes)
                rows.append(dict(condition=condition, budget=budget, method=method,
                                 clean=float(np.mean([r["clean_success30"] for r in data])),
                                 raw=float(np.mean([r["success30"] for r in data]))))
    write_csv(output / "historical_conditions.csv", rows)
    lines = ["# S04 设计依据与边界：S03已完成的五个条件", "",
             "这是已看过的S03数据，不是S04的新初态测试，也不冒充重新完成的一轮训练。",
             "列出停止旧队列时全部完成的五个条件，不删除质量加倍反例。B5尚未齐全，不纳入五条件汇总。",
             "主表针对有限在线预算；同时保留长预算表，防止将样本效率收益误说成长期必胜。"]
    for budget in (BUDGET, 655360):
        lines += ["", f"## {budget}步在线预算", "", "| 条件 | PPO清洁率 | BC→PPO清洁率 | 残差清洁率 | 两者均高于PPO |",
                  "|---|---:|---:|---:|---|"]
        wins = 0
        for condition in CONDITIONS:
            values = {r["method"]: r["clean"] for r in rows if r["condition"] == condition and r["budget"] == budget}
            both = values["guided"] > values["ppo"] and values["residual"] > values["ppo"]
            wins += int(both)
            lines.append(f"| {condition} | {values['ppo']:.2%} | {values['guided']:.2%} | {values['residual']:.2%} | {both} |")
        lines += ["", f"这五个条件中有{wins}/5个两种融合方法均值都高于PPO；这只是所列有限条件的计数，不代表任意任务的比例。"]
    lines += ["", "质量加倍不是可以忽略的异常数据，而是当前参考及控制能力的适用边界。",
              "两种方法的主张应是：已有匹配视频先验时，更少在线探索可获得更高成功率；不是任意预算、任意物体都优于PPO。",
              "完整原始成功率同时保存在CSV；不只看清洁率，也不将小幅均值差自动当成显著收益。"]
    (output / "historical_conditions.md").write_text("\n".join(lines)+"\n")


def report(protocol):
    groups, per_seed, rows = collect(protocol)
    output = OUT / "report"
    output.mkdir(exist_ok=True)
    historical_report(output)
    matrices, aggregate = {}, []
    for method, runs in groups.items():
        data = [r for _, episodes in runs for r in episodes]
        matrix = np.array([[r["clean_success30"] for r in episodes] for _, episodes in runs], float)
        matrices[method] = matrix
        low, high = bootstrap_interval(matrix)
        aggregate.append(dict(method=method, policies=len(runs), episodes=len(data), clean=float(matrix.mean()),
                              ci_low=low, ci_high=high, raw=float(np.mean([r["success30"] for r in data])),
                              full_clean=float(np.mean([r["full_episode_clean_success30"] for r in data])),
                              mean_force_n=float(np.mean([r["substep_force_peak_n"] for r in data])),
                              raw_failures=sum(not r["success30"] for r in data),
                              dirty_successes=sum(r["success30"] and not r["clean_success30"] for r in data)))
    paired = comparisons(matrices)
    stats = dict(complete=True, episodes=len(rows), aggregate=aggregate, comparisons=paired,
                 protocol_sha256=digest(OUT / "protocol.json"), design=protocol["design"])
    (output / "statistics.json").write_text(json.dumps(stats, indent=2)+"\n")
    for name, data in (("episodes", rows), ("per_seed", per_seed), ("aggregate", aggregate), ("paired", paired)):
        write_csv(output / (name+".csv"), data)
    lines = ["# S04 新初态复测结果", "", "定制场景：匹配视频先验、有限在线训练预算。不是全场景优越性结论。",
             "复用S03全部十个种子的163840步权重，使用新的100个共同初态81000..81099；没有重新训练或筛种子。",
             "BC和参考准备成本另计，不把相同在线步数说成相同总成本。", "",
             "| 方法 | 清洁成功率 | 原始成功率 | 全程穿透合格成功率 |", "|---|---:|---:|---:|"]
    for r in aggregate:
        lines.append(f"| {LABELS[r['method']]} | {r['clean']:.2%} | {r['raw']:.2%} | {r['full_clean']:.2%} |")
    lines += ["", "## 两项重点比较", "", "| 比较 | 提升百分点 | 95%差值区间 | Holm-2 p | 本基准支持提升 |",
              "|---|---:|---|---:|---|"]
    for r in paired:
        lines.append(f"| {r['comparison']} | {r['difference_pp']:+.2f} | [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}] | {r['holm_2_p']:.6f} | {r['supported_on_this_benchmark']} |")
    lines += ["", "区间为训练种子/共同初态交叉bootstrap；检验单位为十个配对训练种子，不把每个回合当一次独立训练。",
              "设计是在已观察S03结果后确定的；这是明确的针对性基准。新初态复测不等于全新训练种子上的独立复现。",
              "学习曲线来自16个开发初态，不能当作这100个新测试初态上的逐检查点曲线。",
              "只对有限在线预算下的同任务收益作结论；长预算、错位物体和加重物体上的局限仍保留在S03。"]
    (output / "results.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    colors = ["#687887", "#d38b26", "#169885", "#a39475", "#858585"]
    ax.bar(np.arange(5), [r["clean"]*100 for r in aggregate], color=colors,
           yerr=[[max(0, (r["clean"]-r["ci_low"])*100) for r in aggregate],
                 [max(0, (r["ci_high"]-r["clean"])*100) for r in aggregate]], capsize=4)
    ax.set_xticks(np.arange(5), [LABELS[r["method"]] for r in aggregate], rotation=15, ha="right")
    ax.set(ylabel="Clean 30s success (%)", ylim=(0, 105), title="Prior-friendly benchmark: 163,840 online steps")
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    curve_rows = []
    for method, group in GROUPS.items():
        ys = []
        for seed in SEEDS:
            with (OLD / "confirmation" / group / f"s{seed}" / "metrics.csv").open() as stream:
                data = [r for r in csv.DictReader(stream) if int(r["step"]) <= BUDGET]
            ys.append([float(r["clean_success30"]) for r in data])
            curve_rows.extend(dict(method=method, train_seed=seed, step=int(r["step"]),
                                   clean=float(r["clean_success30"])) for r in data)
        ax.plot([0, 81920, BUDGET], np.mean(ys, axis=0)*100, "o-", label=LABELS[method])
    ax.set(xlabel="Online control steps (prior preparation excluded)", ylabel="Development clean success (%)",
           ylim=(0, 105), title="Learning curves: 16 development starts, ten training seeds")
    ax.legend()
    fig.savefig(output / "learning_curves.png", dpi=150)
    plt.close(fig)
    write_csv(output / "learning_curves.csv", curve_rows)
    (OUT / "audit.json").write_text(json.dumps(dict(complete=True, jobs=41, episodes=len(rows),
        independent_preview_scores=164, paired_initial_states_exact=True,
        protocol_sha256=digest(OUT / "protocol.json"), videos_pending=True), indent=2)+"\n")
    print(json.dumps(stats), flush=True)


def render(protocol):
    output = OUT / "videos"
    output.mkdir(exist_ok=True)
    cases = []
    for job in protocol["jobs"]:
        if job["seed"] != 21001:
            continue
        run = ROOT / job["output"]
        config = json.loads((run / "run_config.json").read_text())
        trace = run / "evaluation/preview_trajectories.npz"
        destination = output / job["method"]
        if not (destination / "report.json").exists():
            if destination.exists():
                raise FileExistsError("Inspect partial render: " + str(destination))
            command = [sys.executable, str(ROOT / "scripts/render_orca_modes.py"), "--model", config["model"],
                       "--trace", str(trace), "--episode", "0", "--output", str(destination),
                       "--azimuth", "225", "--elevation", "-18", "--distance", ".38",
                       "--lookat", ".035", ".025", ".245"]
            with (output / (job["method"]+".log")).open("x") as log:
                subprocess.run(command, cwd=ROOT, check=True, env=dict(os.environ, MUJOCO_GL="egl"),
                               stdout=log, stderr=subprocess.STDOUT)
        rendered = json.loads((destination / "report.json").read_text())
        assert rendered["trace_sha256"] == digest(trace) and rendered["no_physics_resimulation"]
        assert rendered["same_saved_states_both_modes"] and rendered["episode"] == 0
        videos = {}
        for mode in ("stl", "collision"):
            path = destination / (mode+".mp4")
            probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-count_frames", "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate,duration",
                "-of", "json", str(path)], text=True))["streams"][0]
            assert int(probe["nb_read_frames"]) == 800 and probe["r_frame_rate"] == "20/1"
            assert (probe["width"], probe["height"]) == (960, 720) and abs(float(probe["duration"])-40) < .01
            assert min(rendered["pixel_checks"][mode][k] for k in ("first_std", "last_std")) > 5
            videos[mode] = dict(path=str(path.relative_to(OUT)), sha256=digest(path), probe=probe)
        episode = json.loads((run / "evaluation/episodes.json").read_text())[0]
        cases.append(dict(method=job["method"], seed=21001, test_seed=episode["seed"], outcome=episode, videos=videos))
    (output / "manifest.json").write_text(json.dumps(dict(complete=True, cases=cases,
        protocol_sha256=digest(OUT / "protocol.json")), indent=2)+"\n")
    lines = ["# S04 固定案例视频", "", "固定训练种子21001、新测试初态81000，成功和失败均保留；手心视角。",
             "STL与碰撞体使用同一保存轨迹，不重新仿真。", "", "| 方法 | 清洁成功 | STL | 碰撞体 |", "|---|---|---|---|"]
    for c in cases:
        m = c["method"]
        lines.append(f"| {LABELS[m]} | {c['outcome']['clean_success30']} | [播放]({m}/stl.mp4) | [播放]({m}/collision.mp4) |")
    (output / "index.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "report", "render"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
        load_protocol()
        print("Prepared 41 evaluations, no new training")
        return
    protocol = load_protocol()
    if args.stage == "run":
        run(protocol)
        report(protocol)
        render(protocol)
    elif args.stage == "report":
        report(protocol)
    else:
        render(protocol)


if __name__ == "__main__":
    main()
