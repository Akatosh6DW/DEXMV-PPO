"""Matched BC sampling comparison with strict outcome and provenance checks."""

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
    runs = [BASE/"B12_feedback_ranked_recovery", BASE/"B14_feedback_recovery_window"]
    configs = [read(p/"run_config.json") for p in runs]
    ignored = {"output", "sources"}
    assert {k:v for k,v in configs[0].items() if k not in ignored} == {
        k:v for k,v in configs[1].items() if k not in ignored}
    assert configs[0]["bc_updates"] == 4000 and configs[0]["steps"] == 0
    for name in configs[0]["sources"]:
        assert configs[0]["sources"][name] == configs[1]["sources"][name], name
    assert set(configs[1]["sources"])-set(configs[0]["sources"]) == {"run_s09_recovery_window_bc.py"}
    assert read(runs[0]/"recovery_bc_config.json") == read(runs[1]/"recovery_bc_config.json")
    sampling = read(runs[1]/"recovery_window_config.json")
    assert sampling["recovery_samples_per_batch"] == 16 and sampling["recovery_window_frames"] == 80
    assert sampling["late_frames"] == 38440 and sampling["policy_updates"] == 4000
    assert sampling["dataset_sha256"] == configs[0]["dataset_sha256"]
    zero = BASE/"smoke_recovery_window_zero_20261021"
    smoke = BASE/"smoke_ranked_bc_recovery_20261021"
    for name in ("bc_loss.json", "step_000000000/params.pkl", "step_000000000/episodes.json"):
        assert (zero/name).read_bytes() == (smoke/name).read_bytes()
    before = read(BASE/"E101_bc_independent_covariance_control/step_000000000/episodes.json")
    rows, episodes, evidence = [], [], {}
    for run, config in zip(runs, configs):
        for name, sha in config["sources"].items():
            assert digest(run/"sources"/name) == sha
        summary = read(run/"summary.json")
        assert summary["status"] == "completed" and summary["final_step"] == 0
        with (run/"metrics.csv").open() as stream:
            metrics = [{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
        assert len(metrics) == 1 and np.isfinite(list(metrics[0].values())).all()
        folder = run/"step_000000000"
        audit = read(folder/"native_replay_audit.json")
        assert audit["status"] == "verified" and audit["checkpoint_sha256"] == digest(folder/"params.pkl")
        assert audit["trace_sha256"] == digest(folder/"preview_trajectories.npz")
        with np.load(folder/"initial_states.npz") as a, np.load(runs[0]/"step_000000000/initial_states.npz") as b:
            for key in ("seeds", "qpos"):
                np.testing.assert_array_equal(a[key], b[key])
        after = read(folder/"episodes.json")
        episodes.append(after)
        assert len(after) == 64 and metrics[0]["success30"] == np.mean([r["success30"] for r in after])
        for r in after:
            assert r["residual_rms_rad"] == 0 and r["released_rope_force_n"] == 0
            if r["success30"]:
                assert r["hold_sec"] >= 30 and r["final_valid"] and not r["forbidden"]
                assert r["initial_penetration_m"] <= .001 and r["max_released_penetration_m"] <= .003
        losses = read(run/"bc_loss.json")
        rows.append(dict(run=run.name, successes=sum(r["success30"] for r in after), n=64,
            **summary["success30"], **transitions(before, after), hold_sec=metrics[0]["hold_sec"],
            max_depth_mm=metrics[0]["max_penetration_m"]*1000, peak_force_n=metrics[0]["peak_force_n"],
            first_logged_validation_loss=losses[0]["validation_loss"], last_validation_loss=losses[-1]["validation_loss"]))
        evidence[run.name] = {name:digest(run/name) for name in ("run_config.json", "metrics.csv", "summary.json",
            "step_000000000/episodes.json", "step_000000000/native_replay_audit.json")}
    paired = transitions(*episodes)
    out = BASE/"report/recovery_window"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/"overview.csv", rows)
    write_csv(out/"paired_episodes.csv", [dict(seed=a["seed"], mode=a["selected_reference"],
        control=a["success30"], oversampled=b["success30"], control_hold=a["hold_sec"], oversampled_hold=b["hold_sec"])
        for a,b in zip(*episodes)])
    (out/"evidence.json").write_text(json.dumps(dict(status="verified_development_only", evidence=evidence,
        paired=paired, zero_sampler_smoke_exact=True, source_sha256=digest(Path(__file__)),
        scope="One supervised training seed and reused 64 development states; zero PPO updates"), indent=2)+"\n")
    lines = ["# 关键恢复窗口抽样：B14对照B12", "",
        "同一数据、I27源、4000次监督更新和64开发初态，只增加新补救2--4秒样本的抽样频率。",
        "每批256中原期望0.266个改为16.233个；0步PPO，不是完整DAgger或DART。", "",
        "| 组 | 成功 | 相对原BC新增/丢失 | 平均合格秒 | 最大穿透mm | 峰值N |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['successes']}/64 | {r['gained']}/{r['lost']} | "
                     f"{r['hold_sec']:.3f} | {r['max_depth_mm']:.3f} | {r['peak_force_n']:.3f} |")
    lines += ["", f"相对B12，B14新增{paired['gained']}例，丢失{paired['lost']}例。原BC为26/64。",
        "first=last=max为唯一训练后评估，best_step0指PPO步数。不能当作监督训练成功率曲线。",
        "损失的抽样分布已经改变，因此不直接横比两组训练loss；验证样本完全相同。",
        "保存预览完成动作/物理核验；极值含失败，全部失败保留分母。无新测试或多种子确认。"]
    (out/"results.md").write_text("\n".join(lines)+"\n")
    diagnostics = []
    for variant in ("closed", "forced"):
        folder = BASE/f"D47_window_student_{variant}"
        summary = read(folder/"summary.json")
        assert summary["status"] == "completed" and summary["cases"] == 2 and summary["policies"] == 1
        assert summary["source_sha256"] == digest(folder/"probe_s09_recovery_student.py")
        assert summary["teacher_collection_sha256"] == digest(BASE/"D44_teacher_full_pool/results.json")
        for r in read(folder/"results.json"):
            assert r["teacher_forced_observations"] == (variant == "forced")
            assert r["checkpoint_sha256"] == digest(runs[1]/"step_000000000/params.pkl")
            assert r["trace_sha256"] == digest(folder/f"{r['run']}_seed{r['seed']}.npz")
            assert r["native_replay_max_qpos_error"] < 3e-6 and r["native_replay_max_qvel_error"] < 1e-4
            diagnostics.append(dict(variant=variant, **r))
    write_csv(out/"teacher_state_diagnostic.csv", diagnostics)
    (out/"teacher_state_diagnostic.md").write_text("\n".join([
        "# B14同状态恢复诊断", "", "同D46的两个训练查询，前40步教师动作后接手；forced是不可部署的老师观测输入。",
        "所有四条轨迹独立原生回放通过，数据和哈希见CSV。", "",
        "| 种子 | 输入 | 动作RMSE(rad) | 合格秒 | 严格成功 |",
        "|---|---|---:|---:|---|",
        *[f"| {r['seed']} | {r['variant']} | {r['teacher_state_action_rmse_rad']:.5f} | {r['hold_sec']:.2f} | {r['success30']} |" for r in diagnostics],
        "", "动作拟合误差进一步下降，但真实反馈两例均0秒；老师观测下5.25/18.15秒，均未达30秒。",
        "不能将更小拟合误差视为更稳定控制，也不能据两个已见训练案例推断新测试效果。",
    ])+"\n")
    print(json.dumps(dict(rows=rows, paired=paired)), flush=True)


if __name__ == "__main__":
    main()
