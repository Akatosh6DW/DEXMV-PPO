"""Paired evidence for shared versus routed value critics; no hidden model selection."""

import csv
import json
from pathlib import Path
import pickle

import jax
import numpy as np

from analyze_s09_bc_confirmation import write_csv
from analyze_s09_late_experts import transitions
from retarget_s09_reference import digest

BASE = Path(__file__).resolve().parents[1] / "artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"


def main():
    read = lambda p: json.loads(p.read_text())
    runs = [BASE / "E97_residual_shared_critic_control", BASE / "E96_residual_critic_experts"]
    configs = [read(p / "run_config.json") for p in runs]
    protocols = [read(p / "critic_expert_config.json") for p in runs]
    for key in ("seed", "std", "steps", "evals", "learning_rate", "model_sha256", "reference_sha256",
                "method", "eval_start", "eval_count", "eval_batch", "backend", "sources"):
        assert configs[0][key] == configs[1][key], key
    assert configs[0]["steps"] == 40960 and configs[0]["eval_count"] == 64
    assert [p["critic_count"] for p in protocols] == [1, 6]
    assert all(p["initial_critics_identical"] for p in protocols)
    for name in ("env_config",):
        assert {k:v for k,v in configs[0][name].items() if k != "action_variant"} == {
            k:v for k,v in configs[1][name].items() if k != "action_variant"}
    learned = [read(p / "learned_base_config.json") for p in runs]
    assert {k:v for k,v in learned[0].items() if k != "policy_architecture"} == {
        k:v for k,v in learned[1].items() if k != "policy_architecture"}
    assert digest(Path(learned[0]["base_checkpoint"])) == learned[0]["base_sha256"]
    initial_params = [pickle.loads((r / "step_000000000/params.pkl").read_bytes()) for r in runs]
    for a,b in zip(jax.tree_util.tree_leaves(initial_params[0][:2]), jax.tree_util.tree_leaves(initial_params[1][:2])):
        np.testing.assert_array_equal(a,b)
    for critic in initial_params[1][2]["experts"]:
        for a,b in zip(jax.tree_util.tree_leaves(initial_params[0][2]),jax.tree_util.tree_leaves(critic)):
            np.testing.assert_array_equal(a,b)
    before = [read(r / "step_000000000/episodes.json") for r in runs]
    assert before[0] == before[1]
    rows, modes, episodes, evidence, final = [], [], [], {}, []
    for run, protocol in zip(runs, protocols):
        summary = read(run / "summary.json")
        assert summary["status"] == "completed" and summary["final_step"] == 40960
        with (run / "metrics.csv").open() as stream:
            metrics = [{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
        assert [int(r["step"]) for r in metrics] == [0,40960]
        for metric in metrics:
            step = int(metric["step"])
            folder = run / f"step_{step:09d}"
            audit = read(folder / "native_replay_audit.json")
            assert audit["status"] == "verified" and audit["critic_count"] == protocol["critic_count"]
            assert audit["checkpoint_sha256"] == digest(folder / "params.pkl")
            assert audit["trace_sha256"] == digest(folder / "preview_trajectories.npz")
            assert audit["base_sha256"] == learned[0]["base_sha256"]
            with np.load(folder / "initial_states.npz") as a, np.load(runs[0] / "step_000000000/initial_states.npz") as b:
                for key in ("qpos", "seeds"):
                    np.testing.assert_array_equal(a[key], b[key])
            eps = read(folder / "episodes.json")
            assert len(eps) == 64 and metric["success30"] == np.mean([r["success30"] for r in eps])
            episodes.extend(dict(run=run.name, step=step, **r) for r in eps)
            if step == 40960:
                final.append(eps)
        metric = metrics[-1]
        rows.append(dict(run=run.name, critics=protocol["critic_count"], **summary["success30"],
            successes=sum(r["success30"] for r in eps), n=len(eps), **transitions(before[0], eps),
            hold_sec=metric["hold_sec"], max_depth_mm=metric["max_penetration_m"]*1000,
            peak_force_n=metric["peak_force_n"], residual_rms_rad=metric["residual_rms_rad"],
            forbidden_rate=metric["forbidden_rate"], released_rope_force_n=metric["released_rope_force_n"]))
        for mode in range(6):
            a = [r for r in before[0] if r["selected_reference"] == mode]
            b = [r for r in eps if r["selected_reference"] == mode]
            modes.append(dict(run=run.name, mode=mode, n=len(b), successes=sum(r["success30"] for r in b),
                              **transitions(a,b)))
        evidence[run.name] = {name:digest(run/name) for name in ("run_config.json", "metrics.csv", "summary.json",
            "critic_expert_config.json", "step_000040960/episodes.json", "step_000040960/native_replay_audit.json")}
    pair = transitions(final[0],final[1])
    out = BASE / "report/residual_critic_experts"
    out.mkdir(parents=True, exist_ok=True)
    for name,data in (("overview.csv",rows),("orientation.csv",modes),("episodes.csv",episodes)):
        write_csv(out/name,data)
    record = dict(status="completed_development_only", rows=rows, six_vs_shared=pair, evidence=evidence,
                  source_sha256=digest(Path(__file__)), base_sha256=learned[0]["base_sha256"])
    (out/"evidence.json").write_text(json.dumps(record,indent=2)+"\n")
    lines = ["# 共享与朝向分工critic：完整配对结果", "",
        "冻结基础、六个残差actor、奖励、初始动作/价值、40960在线步与64开发初态一致。只改critic结构。", "",
        "| 组 | first | last | max | best_step | 新增/丢失 | 平均合格秒 | 最大穿透mm | 峰值N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['first']:.2%} | {r['last']:.2%} | {r['max']:.2%} | {r['best_step']} | "
                     f"{r['gained']}/{r['lost']} | {r['hold_sec']:.3f} | {r['max_depth_mm']:.3f} | {r['peak_force_n']:.3f} |")
    lines += ["", f"六critic相对共享末步的新增/丢失：{pair['gained']}/{pair['lost']}。",
        "所有四个保存预览经动作组合、完整命令历史和独立原生物理回放审计；并非所有64条都保存了逐帧轨迹。",
        "最大穿透/峰值包含失败案例，不删除异常失败；原40秒/30秒成功判据不变。",
        "六critic共208134参数，对照34689参数，增加容量与路由同时发生，不能独立归因于消除梯度冲突。",
        "一个训练种子、反复使用的开发集，不是新的保留测试或跨种子稳定性结论。",
        "基础继承1310720在线步，每组新增40960步；不把基础已有成功率算作残差增益。",
        "原始逐例结果在episodes.csv；orientation.csv记录分朝向得失；evidence.json绑定源文件。"]
    (out/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(record),flush=True)


if __name__ == "__main__":
    main()
