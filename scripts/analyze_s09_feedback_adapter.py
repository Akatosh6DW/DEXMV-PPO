"""Compare feedback-only BC updates against all-parameter expert BC."""

import csv
import json
from pathlib import Path
import pickle

import numpy as np

from analyze_s09_bc_confirmation import write_csv
from analyze_s09_late_experts import transitions
from analyze_s09_ranked_recovery import BASE
from analyze_s09_recovery_experts import equal_tree
from retarget_s09_reference import digest
from run_s09_feedback_adapter_bc import verify_frozen
from run_s09_mode_experts import expand_parameters


def main():
    read = lambda p: json.loads(p.read_text())
    runs = [BASE / "B15_feedback_recovery_experts", BASE / "B16_feedback_connections_only"]
    configs = [read(p / "run_config.json") for p in runs]
    ignored = {"output", "sources"}
    assert {k: v for k, v in configs[0].items() if k not in ignored} == {
        k: v for k, v in configs[1].items() if k not in ignored}
    bc_configs = [read(p / "recovery_bc_config.json") for p in runs]
    assert bc_configs[0] == bc_configs[1]
    origin = Path(bc_configs[0]["source_checkpoint"])
    assert digest(origin) == bc_configs[0]["source_checkpoint_sha256"]
    before = expand_parameters(pickle.loads(origin.read_bytes()))
    after = pickle.loads((runs[1] / "step_000000000/params.pkl").read_bytes())
    metadata = read(runs[1] / "feedback_adapter_config.json")
    assert metadata["status"] == "verified_parameter_scope" and metadata["bc_updates"] == 4000
    assert verify_frozen(before[1], after[1]) == metadata["changed_parameters"]
    for i in (0, 2):
        equal_tree(before[i], after[i])
    equal_tree(before[1]["experts"][4], after[1]["experts"][4])
    original = read(BASE / "E101_bc_independent_covariance_control/step_000000000/episodes.json")
    rows, modes, raw, evidence, episodes_all = [], [], [], {}, []
    for run, config in zip(runs, configs):
        for name, sha in config["sources"].items():
            assert digest(run / "sources" / name) == sha
        summary = read(run / "summary.json")
        assert summary["status"] == "completed" and summary["final_step"] == 0
        with (run / "metrics.csv").open() as stream:
            metrics = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(stream)]
        assert len(metrics) == 1 and np.isfinite(list(metrics[0].values())).all()
        folder = run / "step_000000000"
        audit = read(folder / "native_replay_audit.json")
        assert audit["status"] == "verified" and audit["checkpoint_sha256"] == digest(folder / "params.pkl")
        assert audit["trace_sha256"] == digest(folder / "preview_trajectories.npz")
        with np.load(folder / "initial_states.npz") as a, np.load(runs[0] / "step_000000000/initial_states.npz") as b:
            for key in ("seeds", "qpos"):
                np.testing.assert_array_equal(a[key], b[key])
        episodes = read(folder / "episodes.json")
        episodes_all.append(episodes)
        assert len(episodes) == 64
        assert metrics[0]["success30"] == np.mean([r["success30"] for r in episodes])
        for r in episodes:
            assert r["released_rope_force_n"] == r["residual_rms_rad"] == 0
            if r["success30"]:
                assert r["hold_sec"] >= 30 and r["final_valid"] and not r["forbidden"]
                assert r["initial_penetration_m"] <= .001 and r["max_released_penetration_m"] <= .003
        losses = read(run / "bc_loss.json")
        assert losses[-1]["step"] == 4000
        rows.append(dict(run=run.name, successes=sum(r["success30"] for r in episodes),
            **summary["success30"], **transitions(original, episodes), hold_sec=metrics[0]["hold_sec"],
            max_depth_mm=1000 * metrics[0]["max_penetration_m"], peak_force_n=metrics[0]["peak_force_n"],
            train_loss=losses[-1]["train_loss"], validation_loss=losses[-1]["validation_loss"]))
        raw.extend(dict(run=run.name, **r) for r in episodes)
        for mode in range(6):
            part = [r for r in episodes if r["selected_reference"] == mode]
            modes.append(dict(run=run.name, mode=mode, n=len(part), successes=sum(r["success30"] for r in part)))
        evidence[run.name] = {p: digest(run / p) for p in ("run_config.json", "bc_loss.json", "metrics.csv",
            "step_000000000/params.pkl", "step_000000000/episodes.json", "step_000000000/native_replay_audit.json")}
    paired = transitions(*episodes_all)
    out = BASE / "report/feedback_adapter"
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("overview.csv", rows), ("orientation.csv", modes), ("episodes.csv", raw)):
        write_csv(out / name, data)
    (out / "evidence.json").write_text(json.dumps(dict(status="verified_development_only", evidence=evidence,
        paired=paired, parameter_scope_verified=True, inherited_bc_updates=40000, additional_bc_updates=4000,
        ppo_updates=0, source_sha256=digest(Path(__file__)), original_checkpoint_sha256=digest(origin),
        limitations="One BC seed, reused dev states; freezing weights does not freeze actions"), indent=2) + "\n")
    lines = ["# 只学新增反馈连接的BC对照", "", "同I27源、D45示范、4000监督更新、六专家网络。只有可更新参数范围不同。",
        "原BC同开发集26/64；两组均新增0步PPO，不使用更新后最优检查点挑选。", "",
        "| 组 | 成功/64 | 相对原BC新增/丢失 | 平均合格秒 | 最大穿透mm | 峰值N | 验证损失 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['successes']} | {r['gained']}/{r['lost']} | {r['hold_sec']:.3f} | "
                     f"{r['max_depth_mm']:.3f} | {r['peak_force_n']:.3f} | {r['validation_loss']:.6f} |")
    lines += ["", f"仅反馈连接相对全参数组新增{paired['gained']}例、丢失{paired['lost']}例。",
        "每组只有一次训练后评估，first=last=max、best_step0，不是BC成功率学习曲线。",
        "原参数/critic/normalizer及无样本第4类专家不变；保存预览动作与原生物理核验通过。",
        "参数保护不保证输出保护；监督拟合更低不等于闭环恢复更好。结果不是新初态或多训练重复确认。"]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    diagnostics = []
    for prefix, run in (("D48_expert", runs[0]), ("D52_adapter", runs[1])):
        for mode in ("closed", "forced"):
            folder = BASE / f"{prefix}_student_{mode}"
            summary = read(folder / "summary.json")
            assert summary["status"] == "completed" and summary["cases"] == 2
            assert summary["prefix_control_steps"] == 40
            assert summary["teacher_forced_observations"] == (mode == "forced")
            assert summary["teacher_collection_sha256"] == digest(BASE / "D44_teacher_full_pool/results.json")
            assert summary["source_sha256"] == digest(folder / "probe_s09_recovery_student.py")
            for r in read(folder / "results.json"):
                assert r["checkpoint_sha256"] == digest(run / "step_000000000/params.pkl")
                assert r["trace_sha256"] == digest(folder / f"{run.name}_seed{r['seed']}.npz")
                assert r["native_replay_max_qpos_error"] < 3e-6 and r["native_replay_max_qvel_error"] < 1e-4
                diagnostics.append(dict(diagnostic=folder.name, **r))
    write_csv(out / "recovery_diagnostics.csv", diagnostics)
    (out / "recovery_diagnostics.md").write_text("\n".join([
        "# 同状态接手：B15与B16", "", "两条已见训练状态，同老师前40步，全部执行轨迹独立原生回放通过。",
        "forced输入老师观测仅为诊断，不是部署控制。严格成功仍为连续30秒且终点合格。", "",
        "| 学生 | 输入 | 种子 | 合格秒 | 成功 | 老师状态动作RMSE rad |",
        "|---|---|---:|---:|---|---:|",
        *[f"| {r['run']} | {'老师' if r['teacher_forced_observations'] else '真实'} | {r['seed']} | "
          f"{r['hold_sec']:.3f} | {r['success30']} | {r['teacher_state_action_rmse_rad']:.6f} |" for r in diagnostics],
        "", "开发成功数提高不等于学会了这两条补救，训练案例诊断不等于未见恢复泛化。",
    ]) + "\n")
    print(json.dumps(dict(rows=rows, paired=paired)), flush=True)


if __name__ == "__main__":
    main()
