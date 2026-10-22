"""Audit matched short PPO runs; no held-out or multi-training-seed claims."""

import csv
import json
from pathlib import Path

import numpy as np

from analyze_s09_bc_confirmation import write_csv
from analyze_s09_late_experts import transitions
from retarget_s09_reference import digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"


def main():
    read = lambda p: json.loads(p.read_text())
    runs = [BASE/"E101_bc_independent_covariance_control", BASE/"E100_bc_correlated_exploration"]
    configs = [read(p/"run_config.json") for p in runs]
    for key in ("checkpoint_sha256","seed","std","steps","evals","learning_rate",
                "model_sha256","reference_sha256","env_config","method","eval_start",
                "eval_count","eval_batch","backend","sources"):
        assert configs[0][key] == configs[1][key], key
    assert configs[0]["steps"] == 40960 and configs[0]["eval_count"] == 64
    for name in ("constraint_config.json", "improvement_config.json"):
        assert read(runs[0]/name) == read(runs[1]/name), name
    protocols = [read(p/"correlated_exploration_config.json") for p in runs]
    assert [p["correlation_strength"] for p in protocols] == [0,.5]
    changed = {"correlation_strength","correlation_matrix","eigenvalues"}
    assert {k:v for k,v in protocols[0].items() if k not in changed} == {
        k:v for k,v in protocols[1].items() if k not in changed}
    initial = [read(p/"step_000000000/episodes.json") for p in runs]
    assert initial[0] == initial[1] and sum(r["success30"] for r in initial[0]) == 26
    assert digest(runs[0]/"step_000000000/params.pkl") == digest(runs[1]/"step_000000000/params.pkl")
    rows, modes, evidence, last = [], [], {}, []
    for run, config in zip(runs, configs):
        for name, sha in config["sources"].items():
            assert digest(run/"sources"/name) == sha
        summary = read(run/"summary.json")
        assert summary["status"] == "completed" and summary["final_step"] == 40960
        with (run/"metrics.csv").open() as stream:
            metrics = [{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
        assert [int(r["step"]) for r in metrics] == [0,40960]
        assert all(np.isfinite(list(r.values())).all() for r in metrics)
        after = read(run/"step_000040960/episodes.json")
        last.append(after)
        for step in (0,40960):
            folder = run/f"step_{step:09d}"
            audit = read(folder/"native_replay_audit.json")
            assert audit["status"] == "verified"
            assert audit["checkpoint_sha256"] == digest(folder/"params.pkl")
            assert audit["trace_sha256"] == digest(folder/"preview_trajectories.npz")
            with np.load(folder/"initial_states.npz") as a, np.load(runs[0]/"step_000000000/initial_states.npz") as b:
                for key in ("qpos","seeds"):
                    np.testing.assert_array_equal(a[key],b[key])
        metric = metrics[-1]
        assert metric["success30"] == np.mean([r["success30"] for r in after])
        for e in after:
            assert e["released_rope_force_n"] == 0 and e["residual_rms_rad"] == 0
            if e["success30"]:
                assert e["hold_sec"] >= 30 and e["final_valid"] and not e["forbidden"]
                assert e["max_released_penetration_m"] <= .003 and e["initial_penetration_m"] <= .001
        rows.append(dict(run=run.name,**summary["success30"],successes=sum(r["success30"] for r in after),
            n=len(after),**transitions(initial[0],after),hold_sec=metric["hold_sec"],
            max_depth_mm=metric["max_penetration_m"]*1000,peak_force_n=metric["peak_force_n"],
            forbidden_rate=metric["forbidden_rate"],released_rope_force_n=metric["released_rope_force_n"]))
        for mode in range(6):
            old = [r for r in initial[0] if r["selected_reference"] == mode]
            new = [r for r in after if r["selected_reference"] == mode]
            modes.append(dict(run=run.name,mode=mode,n=len(new),successes=sum(r["success30"] for r in new),
                              **transitions(old,new)))
        evidence[run.name] = {name:digest(run/name) for name in ("run_config.json","metrics.csv","summary.json",
            "correlated_exploration_config.json","step_000040960/episodes.json","step_000040960/native_replay_audit.json")}
    out = BASE/"report/correlated_exploration"
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"overview.csv",rows)
    write_csv(out/"orientation.csv",modes)
    paired = transitions(*last)
    record = dict(status="completed_development_only",evidence=evidence,source_sha256=digest(Path(__file__)),
        initial_episodes_and_parameters_identical=True,correlated_vs_control=paired,training_seed=14102,
        scope="64 reused development initial states, one training seed, not full Lattice or residual")
    (out/"evidence.json").write_text(json.dumps(record,indent=2)+"\n")
    lines = ["# BC+PPO关节相关探索：匹配短训练", "",
        "同一I27起点26/64，40960步，seed14102，原64开发初态。仅改变固定相关矩阵强度0与0.5。",
        "两组使用相同多变量高斯实现；部署仍是单个完整目标网络。不是残差，不是完整Lattice。", "",
        "| 组 | 首次 | 末次 | 最高 | best step | 新增/丢失 | 平均合格秒 | 最大穿透mm | 峰值N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['first']:.2%} | {r['last']:.2%} | {r['max']:.2%} | "
                     f"{r['best_step']} | {r['gained']}/{r['lost']} | {r['hold_sec']:.3f} | "
                     f"{r['max_depth_mm']:.3f} | {r['peak_force_n']:.3f} |")
    lines += ["",f"相关组相对同预算控制新增{paired['gained']}例、丢失{paired['lost']}例。",
        "这是单训练种子的开发结果，不是统计确认。全部失败保留，接触极值含失败。",
        "0/40960保存预览完成权重动作及原生回放审计，不冒充全部64条轨迹均独立审计。",
        "初始化边际标准差不变；相关性会改变联合熵、KL及更新方向，不能将结果全归因于物理协调。"]
    (out/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(rows=rows,paired=paired)),flush=True)


if __name__ == "__main__":
    main()
