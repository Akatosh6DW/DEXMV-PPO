"""Verify the matched B12/B13 data experiment; never infer PPO gains."""

import csv
import json
from pathlib import Path

import numpy as np

from analyze_s09_bc_confirmation import write_csv
from analyze_s09_late_experts import transitions
from retarget_s09_reference import digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"


def main():
    read = lambda p: json.loads(p.read_text())
    runs = [BASE / "B13_feedback_ranked_control", BASE / "B12_feedback_ranked_recovery"]
    configs = [read(p / "run_config.json") for p in runs]
    ignored = {"output", "dataset", "dataset_sha256"}
    assert {k: v for k, v in configs[0].items() if k not in ignored} == {
        k: v for k, v in configs[1].items() if k not in ignored}
    refinements = [read(p / "recovery_bc_config.json") for p in runs]
    assert refinements[0] == refinements[1]
    assert refinements[0]["bc_updates"] == 4000
    assert refinements[0]["ppo_steps"] == 0 and not refinements[0]["phase_only"]
    assert digest(Path(refinements[0]["source_checkpoint"])) == refinements[0]["source_checkpoint_sha256"]
    assert configs[0]["eval_count"] == 64 and configs[0]["eval_start"] == 130000
    datasets = [Path(c["dataset"]) for c in configs]
    manifests = [read(p.parent / "dataset_manifest.json") for p in datasets]
    assert [m["train_episodes"] for m in manifests] == [48, 50]
    for p, c, m in zip(datasets, configs, manifests):
        assert digest(p) == c["dataset_sha256"] == m["dataset_sha256"]
    with np.load(datasets[0]) as old, np.load(datasets[1]) as new:
        for key in ("obs", "action", "active"):
            np.testing.assert_array_equal(old[key][:48], new[key][:48])
            np.testing.assert_array_equal(old[key][48:], new[key][50:])
        assert not new["active"][48:50, :40].any()
    collected = read(datasets[1].parent / "summary.json")
    assert collected["status"] == "completed" and collected["validation_episodes"] == 14
    assert [r["seed"] for r in collected["accepted"]] == [90531, 90540]

    baseline_folder = BASE / "E101_bc_independent_covariance_control/step_000000000"
    before = read(baseline_folder / "episodes.json")
    assert len(before) == 64 and sum(r["success30"] for r in before) == 26
    rows, all_episodes, modes, evidence = [], [], [], {}
    for run, config in zip(runs, configs):
        for name, sha in config["sources"].items():
            assert digest(run / "sources" / name) == sha
        summary = read(run / "summary.json")
        assert summary["status"] == "completed" and summary["final_step"] == 0
        with (run / "metrics.csv").open() as stream:
            metrics = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(stream)]
        assert len(metrics) == 1 and metrics[0]["step"] == 0
        assert np.isfinite(list(metrics[0].values())).all()
        folder = run / "step_000000000"
        after = read(folder / "episodes.json")
        all_episodes.append(after)
        audit = read(folder / "native_replay_audit.json")
        assert audit["status"] == "verified"
        assert audit["checkpoint_sha256"] == digest(folder / "params.pkl")
        assert audit["trace_sha256"] == digest(folder / "preview_trajectories.npz")
        with np.load(folder / "initial_states.npz") as a, np.load(baseline_folder / "initial_states.npz") as b:
            for key in ("qpos", "seeds"):
                np.testing.assert_array_equal(a[key], b[key])
        assert metrics[0]["success30"] == np.mean([r["success30"] for r in after])
        for episode in after:
            assert episode["residual_rms_rad"] == 0 and episode["released_rope_force_n"] == 0
            if episode["success30"]:
                assert episode["hold_sec"] >= 30 and episode["final_valid"] and not episode["forbidden"]
                assert episode["initial_penetration_m"] <= .001
                assert episode["max_released_penetration_m"] <= .003
        losses = read(run / "bc_loss.json")
        assert losses[-1]["step"] == 4000
        rows.append(dict(run=run.name, n=len(after), successes=sum(r["success30"] for r in after),
            **summary["success30"], **transitions(before, after), hold_sec=metrics[0]["hold_sec"],
            max_depth_mm=metrics[0]["max_penetration_m"]*1000, peak_force_n=metrics[0]["peak_force_n"],
            first_logged_train_loss=losses[0]["train_loss"], last_train_loss=losses[-1]["train_loss"],
            first_logged_validation_loss=losses[0]["validation_loss"], last_validation_loss=losses[-1]["validation_loss"]))
        for mode in range(6):
            left = [r for r in before if r["selected_reference"] == mode]
            right = [r for r in after if r["selected_reference"] == mode]
            modes.append(dict(run=run.name, mode=mode, n=len(right),
                successes=sum(r["success30"] for r in right), **transitions(left, right)))
        evidence[run.name] = {name: digest(run/name) for name in (
            "run_config.json", "recovery_bc_config.json", "metrics.csv", "summary.json", "bc_loss.json",
            "step_000000000/episodes.json", "step_000000000/native_replay_audit.json")}
    paired = transitions(*all_episodes)
    out = BASE / "report/ranked_recovery"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/"overview.csv", rows)
    write_csv(out/"orientation.csv", modes)
    write_csv(out/"paired_episodes.csv", [dict(seed=a["seed"], mode=a["selected_reference"],
        original_bc=old["success30"], control=a["success30"], new_data=b["success30"],
        control_hold=a["hold_sec"], new_data_hold=b["hold_sec"])
        for old, a, b in zip(before, *all_episodes)])
    record = dict(status="verified_development_only", evidence=evidence, paired=paired,
        source_sha256=digest(Path(__file__)), original_data_preserved=True, ppo_steps=0,
        dataset_summary_sha256=digest(datasets[1].parent/"summary.json"),
        limitation="One BC training seed; reused development states; successful teacher is not successful student")
    (out/"evidence.json").write_text(json.dumps(record, indent=2)+"\n")
    lines = ["# 补救老师改善能否传给BC学生", "",
        "同一I27初始化、4000次BC更新、Adam1e-4、seed13112，0步PPO。",
        "B13保留48条训练示范；B12只追加两条完整时长筛选的补救。原14条验证逐元素相同。",
        "评估为反复使用的64开发初态，不是独立确认。原BC成功26/64。", "",
        "| 组 | 成功 | 相对原BC新增/丢失 | 平均合格秒 | 最大穿透mm | 峰值N |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['successes']}/64 | {r['gained']}/{r['lost']} | "
                     f"{r['hold_sec']:.3f} | {r['max_depth_mm']:.3f} | {r['peak_force_n']:.3f} |")
    lines += ["", f"新数据相对匹配控制新增{paired['gained']}例、丢失{paired['lost']}例。",
        "只有训练后一个评估点，所以first=last=max，best_step=0是PPO步数，不能画成BC成功率学习曲线。",
        "bc_loss.json从第1次更新后开始记录，不冒充更新前损失；训练损失为随机小批次，验证损失在固定验证集上计算。",
        "保存的四条预览完成动作和原生物理回放核验，不代表所有64条完整轨迹均独立审计。",
        "老师补救成功与学生成功分开报告；接触极值含失败，不将大力挤压算优质抓取。"]
    (out/"results.md").write_text("\n".join(lines)+"\n")
    diagnostics = []
    for variant in ("closed", "forced"):
        folder = BASE/f"D46_ranked_student_{variant}"
        summary = read(folder/"summary.json")
        assert summary["status"] == "completed" and summary["cases"] == 2 and summary["policies"] == 3
        assert summary["ranked_teacher_input"] and summary["prefix_control_steps"] == 40
        assert summary["teacher_collection_sha256"] == digest(BASE/"D44_teacher_full_pool/results.json")
        assert summary["source_sha256"] == digest(folder/"probe_s09_recovery_student.py")
        for row in read(folder/"results.json"):
            assert row["teacher_forced_observations"] == (variant == "forced")
            assert row["native_replay_max_qpos_error"] < 3e-6 and row["native_replay_max_qvel_error"] < 1e-4
            assert row["trace_sha256"] == digest(folder/f"{row['run']}_seed{row['seed']}.npz")
            assert row["teacher_trace_sha256"] == digest(BASE/f"D44_teacher_full_pool/seed{row['seed']}/canonical_trace.npz")
            diagnostics.append(dict(variant=variant, **row))
    write_csv(out/"teacher_state_diagnostic.csv", diagnostics)
    diagnostic_lines = ["# 同一老师状态接手：动作拟合与闭环执行", "",
        "三个模型共用老师前40步，再由学生输出全部动作。forced始终输入记录的老师观测，仅作机制诊断，不能部署。",
        "两个状态都是新增示范的训练案例，不是新测试。全部12条执行轨迹通过独立原生回放。", "",
        "| 模型 | 种子 | 输入 | 动作RMSE(rad) | 合格保持秒 | 成功 |",
        "|---|---:|---|---:|---:|---|"]
    for r in diagnostics:
        diagnostic_lines.append(f"| {r['run']} | {r['seed']} | {r['variant']} | "
            f"{r['teacher_state_action_rmse_rad']:.5f} | {r['hold_sec']:.2f} | {r['success30']} |")
    diagnostic_lines += ["", "B12在两条老师轨迹上的拟合误差比B13小，但真实反馈和老师观测输入均未达标。",
        "不能把失败单独归结为状态分布偏离；当前拟合动作即使配合老师观测也不足以成功。",
        "这是两个案例的机制证据，不是全局因果定论，也没有证明只增加监督更新就能修复。"]
    (out/"teacher_state_diagnostic.md").write_text("\n".join(diagnostic_lines)+"\n")
    print(json.dumps(dict(rows=rows, paired=paired)), flush=True)


if __name__ == "__main__":
    main()
