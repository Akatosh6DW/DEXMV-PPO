"""Separate PPO improvements from BC specialization and runtime differences."""

import csv
import json
from pathlib import Path

import numpy as np

from analyze_s09_bc_confirmation import write_csv
from analyze_s09_late_experts import transitions
from analyze_s09_ranked_recovery import BASE
from retarget_s09_reference import digest


def main():
    read = lambda p: json.loads(p.read_text())
    runs = [BASE/"E102_feedback_expert_ppo", BASE/"E103_feedback_bc_residual"]
    bc = BASE/"B15_feedback_recovery_experts"
    source_hash = digest(bc/"step_000000000/params.pkl")
    original = read(bc/"step_000000000/episodes.json")
    configs = [read(p/"run_config.json") for p in runs]
    for key in ("model_sha256", "reference_sha256", "seed", "std", "steps", "evals", "eval_start",
                "eval_count", "eval_batch", "learning_rate", "backend", "orientation"):
        assert configs[0][key] == configs[1][key], key
    assert configs[0]["checkpoint_sha256"] == source_hash
    assert configs[0]["steps"] == 40960 and configs[0]["eval_count"] == 64
    protocol = read(runs[1]/"learned_base_config.json")
    assert protocol["base_sha256"] == source_hash and protocol["base_is_pure_bc"]
    assert not protocol["base_parameters_in_optimizer"]
    assert protocol["critic_learning_rate"] == read(runs[0]/"constraint_config.json")["critic_learning_rate"]
    rows, curves, modes, raw, evidence, initials, finals = [], [], [], [], {}, [], []
    for run, config in zip(runs, configs):
        for name, sha in config["sources"].items():
            assert digest(run/"sources"/name) == sha
        summary = read(run/"summary.json")
        assert summary["status"] == "completed" and summary["final_step"] == 40960
        with (run/"metrics.csv").open() as stream:
            metrics = [{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
        assert [r["step"] for r in metrics] == [0,40960]
        by_step = {}
        for metric in metrics:
            assert np.isfinite(list(metric.values())).all()
            step = int(metric["step"])
            folder = run/f"step_{step:09d}"
            audit = read(folder/"native_replay_audit.json")
            assert audit["status"] == "verified" and audit["checkpoint_sha256"] == digest(folder/"params.pkl")
            assert audit["trace_sha256"] == digest(folder/"preview_trajectories.npz")
            with np.load(folder/"initial_states.npz") as a, np.load(bc/"step_000000000/initial_states.npz") as b:
                for key in ("qpos", "seeds"):
                    np.testing.assert_array_equal(a[key], b[key])
            eps = read(folder/"episodes.json")
            by_step[step] = eps
            assert len(eps) == 64 and np.mean([r["success30"] for r in eps]) == metric["success30"]
            for r in eps:
                assert r["released_rope_force_n"] == 0
                if r["success30"]:
                    assert r["hold_sec"] >= 30 and r["final_valid"] and not r["forbidden"]
                    assert r["initial_penetration_m"] <= .001 and r["max_released_penetration_m"] <= .003
            if step == 0:
                assert metric["residual_rms_rad"] == 0
            curves.append(dict(run=run.name, **metric))
            raw.extend(dict(run=run.name, step=step, **r) for r in eps)
            for mode in range(6):
                part = [r for r in eps if r["selected_reference"] == mode]
                modes.append(dict(run=run.name, step=step, mode=mode, n=len(part), successes=sum(r["success30"] for r in part)))
            evidence[f"{run.name}/{step}"] = dict(episodes_sha256=digest(folder/"episodes.json"),
                audit_sha256=digest(folder/"native_replay_audit.json"), checkpoint_sha256=digest(folder/"params.pkl"))
        initials.append(by_step[0]); finals.append(by_step[40960])
        rows.append(dict(run=run.name, **summary["success30"],
            initial_successes=sum(r["success30"] for r in by_step[0]), final_successes=sum(r["success30"] for r in by_step[40960]),
            **transitions(by_step[0],by_step[40960]), final_hold_sec=metrics[-1]["hold_sec"],
            max_depth_mm=1000*metrics[-1]["max_penetration_m"], peak_force_n=metrics[-1]["peak_force_n"],
            residual_rms_rad=metrics[-1]["residual_rms_rad"]))
    differences = [dict(run=run.name, **transitions(original, eps),
        differing_episode_fields={key:sum(a[key]!=b[key] for a,b in zip(original,eps)) for key in original[0]})
        for run,eps in zip(runs,initials)]
    out = BASE/"report/feedback_bc_ppo"
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("overview.csv",rows), ("curve.csv",curves), ("orientation.csv",modes), ("episodes.csv",raw)):
        write_csv(out/name,data)
    record = dict(status="verified_development_only", source_sha256=digest(Path(__file__)), evidence=evidence,
        bc_checkpoint_sha256=source_hash, initial_runtime_differences=differences,
        residual_vs_full_final=transitions(*finals), inherited_online_steps=0,
        inherited_bc_updates=44000, additional_online_steps_per_arm=40960,
        limitations="One PPO seed and reused dev states; different actor capacity and physical exploration; use each measured start")
    (out/"evidence.json").write_text(json.dumps(record,indent=2)+"\n")
    lines = ["# 反馈BC接PPO：完整动作与执行级残差", "",
        "同一个B15反馈BC基座，新增各40960在线步。B15自身为17/64，原I27为26/64。",
        "原I27监督40000更新、B15新增4000更新；没有继承PPO权重。额外示范有教师仿真成本，不能当无成本。", "",
        "| 组 | 初始 | 末步 | 最高 | best step | 新增/丢失 | 末步合格秒 | 最大穿透mm | 峰值N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['initial_successes']}/64 | {r['final_successes']}/64 | {r['max']:.2%} | "
            f"{r['best_step']} | {r['gained']}/{r['lost']} | {r['final_hold_sec']:.3f} | {r['max_depth_mm']:.3f} | {r['peak_force_n']:.3f} |")
    lines += ["", "零残差的理论动作恒等不意味着改变计算图后每条接触轨迹逐位一致；初始化差异完整保存在evidence.json。",
        "收益按每组自己的实测0步比较，不把初始化差异算作训练提升。两组不是等参数量或等物理噪声对照。",
        "保存预览均完成权重动作与独立物理核验，残差额外核验冻结基础、合成及完整动作历史。",
        "单训练种子、反复使用的64开发初态，不是未见测试或跨训练重复的稳定性证明。"]
    (out/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(rows=rows, initial_differences=differences)),flush=True)


if __name__ == "__main__":
    main()
