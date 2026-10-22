"""Paired, unchanged-task development comparison of random-disturbance PPO."""

import csv
import json
from pathlib import Path

import numpy as np

from analyze_s09_late_experts import transitions
from retarget_s09_reference import digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"


def main():
    names = ["E94_bc_disturbance_control", "E95_bc_disturbance_training"]
    read = lambda p: json.loads(p.read_text())
    runs = [BASE/n for n in names]
    configs = [read(p/"run_config.json") for p in runs]
    for key in ("checkpoint_sha256","seed","std","steps","evals","learning_rate",
                "model_sha256","reference_sha256","env_config","method","eval_start",
                "eval_count","eval_batch","backend","sources"):
        assert configs[0][key] == configs[1][key], key
    assert configs[0]["steps"] == 40960 and configs[0]["eval_count"] == 64
    for name in ("constraint_config.json", "improvement_config.json"):
        assert read(runs[0]/name) == read(runs[1]/name), name
    protocols = [read(p/"disturbance_training_config.json") for p in runs]
    assert [c["probability"] for c in protocols] == [0,.5]
    assert {k:v for k,v in protocols[0].items() if k!="probability"} == {
        k:v for k,v in protocols[1].items() if k!="probability"}
    initial = [read(p/"step_000000000/episodes.json") for p in runs]
    assert initial[0] == initial[1]
    assert sum(r["success30"] for r in initial[0]) == 26
    rows, modes, evidence, last = [], [], {}, []
    for run in runs:
        summary = read(run/"summary.json")
        assert summary["status"] == "completed" and summary["final_step"] == 40960
        with (run/"metrics.csv").open() as stream:
            metrics = [{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
        assert [int(r["step"]) for r in metrics] == [0,40960]
        after = read(run/"step_000040960/episodes.json")
        last.append(after)
        for step in (0,40960):
            folder = run/f"step_{step:09d}"
            audit = read(folder/"native_replay_audit.json")
            assert audit["status"] == "verified"
            assert audit["checkpoint_sha256"] == digest(folder/"params.pkl")
            assert audit["trace_sha256"] == digest(folder/"preview_trajectories.npz")
            with np.load(folder/"initial_states.npz") as a, np.load(runs[0]/"step_000000000/initial_states.npz") as b:
                for key in ("qpos", "seeds"):
                    np.testing.assert_array_equal(a[key], b[key])
        metric = metrics[-1]
        assert metric["success30"] == np.mean([r["success30"] for r in after])
        rows.append(dict(run=run.name, **summary["success30"], successes=sum(r["success30"] for r in after),
            n=len(after), **transitions(initial[0],after), hold_sec=metric["hold_sec"],
            max_depth_mm=metric["max_penetration_m"]*1000, peak_force_n=metric["peak_force_n"],
            forbidden_rate=metric["forbidden_rate"], released_rope_force_n=metric["released_rope_force_n"]))
        for mode in range(6):
            before_part = [r for r in initial[0] if r["selected_reference"]==mode]
            after_part = [r for r in after if r["selected_reference"]==mode]
            modes.append(dict(run=run.name, mode=mode, n=len(after_part),
                successes=sum(r["success30"] for r in after_part), **transitions(before_part,after_part)))
        evidence[run.name] = {name:digest(run/name) for name in (
            "run_config.json", "metrics.csv", "summary.json", "disturbance_training_config.json",
            "step_000040960/episodes.json", "step_000040960/native_replay_audit.json")}
    out = BASE/"report/disturbance_training"
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("overview.csv",rows),("orientation.csv",modes)):
        with (out/name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    paired = transitions(*last)
    (out/"evidence.json").write_text(json.dumps(dict(status="completed_development_only",
        evidence=evidence, source_sha256=digest(Path(__file__)),
        initial_episodes_identical=True, perturbed_vs_control=paired),indent=2)+"\n")
    lines = ["# BC+PPO训练扰动：原任务匹配评估", "",
        "同一I27起点26/64，相同40960步、seed14102和原64开发初态。只改变训练中第2秒施加0.02m/s速度扰动的概率；评估不加扰动。",
        "不是新未见测试，不是鲁棒性测试全集，不将单次开发差异称作统计确认。", "",
        "| 组 | 首次 | 末次 | 最高 | best step | 新增/丢失 | 平均合格秒 | 最大穿透mm | 峰值N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['first']:.2%} | {r['last']:.2%} | {r['max']:.2%} | {r['best_step']} | {r['gained']}/{r['lost']} | {r['hold_sec']:.3f} | {r['max_depth_mm']:.3f} | {r['peak_force_n']:.3f} |")
    lines += ["",f"扰动组相对同预算控制新增{paired['gained']}例、丢失{paired['lost']}例。",
        "所有失败都计入统计，最大穿透和峰值包含失败案例。0/40960两阶段的保存预览完成动作与原生回放核验，不冒充全部未保存轨迹均审计。",
        "原任务成功门槛、模型、奖励与评价初态均不变。完整逐例数据在两组step目录；overview.csv、orientation.csv和evidence.json提供可复核统计。"]
    (out/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(rows=rows, paired=paired)))


if __name__ == "__main__":
    main()
