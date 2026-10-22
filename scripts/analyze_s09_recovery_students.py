"""Audited, paired development summary for feedback-BC refinement experiments."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_s09_late_experts import transitions
from retarget_s09_reference import digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=["B02_feedback_original_data", "B03_feedback_recovery_data", "B04_phase_recovery_data"])
    parser.add_argument("--output", type=Path, default=BASE/"report/recovery_students")
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text())
    control = BASE/"E27_bc_fast_critic_s14102"
    baseline_cfg = read(control/"run_config.json")
    baseline = control/"step_000000000"
    before = read(baseline/"episodes.json")
    assert len(before) == 64 and sum(x["success30"] for x in before) == 26
    rows, modes, evidence = [], [], {}
    for name in args.runs:
        run = BASE/name
        cfg = read(run/"run_config.json")
        refinement = read(run/"recovery_bc_config.json")
        summary = read(run/"summary.json")
        folder = run/"step_000000000"
        for key in ("model_sha256", "reference_sha256", "env_config", "method", "eval_start", "eval_count", "eval_batch", "backend"):
            assert cfg[key] == baseline_cfg[key], key
        assert cfg["steps"] == 0 and cfg["seed"] == 13112
        assert refinement["bc_updates"] == 4000 and refinement["bc_learning_rate"] == 1e-4
        assert refinement["source_checkpoint_sha256"] == baseline_cfg["checkpoint_sha256"]
        assert digest(Path(refinement["source_checkpoint"])) == refinement["source_checkpoint_sha256"]
        assert digest(Path(cfg["dataset"])) == cfg["dataset_sha256"]
        assert summary["status"] == "completed"
        audit = read(folder/"native_replay_audit.json")
        assert audit["status"] == "verified" and audit["normalized_action_tolerance"] == 1e-5
        assert audit["checkpoint_sha256"] == digest(folder/"params.pkl")
        assert audit["trace_sha256"] == digest(folder/"preview_trajectories.npz")
        with np.load(folder/"initial_states.npz") as a, np.load(baseline/"initial_states.npz") as b:
            for key in ("seeds", "qpos"):
                np.testing.assert_array_equal(a[key], b[key])
        after = read(folder/"episodes.json")
        row = summary["rows"][0]
        assert row["success30"] == np.mean([r["success30"] for r in after])
        losses = read(run/"bc_loss.json")
        assert losses[-1]["step"] == 4000
        rows.append(dict(run=name, successful_episodes=sum(x["success30"] for x in after),
            episodes=len(after), success30=row["success30"], hold_sec=row["hold_sec"],
            max_depth_mm=row["max_penetration_m"]*1000, peak_force_n=row["peak_force_n"],
            forbidden_rate=row["forbidden_rate"], released_rope_force_n=row["released_rope_force_n"],
            **transitions(before, after), train_loss=losses[-1]["train_loss"], validation_loss=losses[-1]["validation_loss"]))
        for mode in range(6):
            part = [x for x in after if x["selected_reference"] == mode]
            old = [x for x in before if x["selected_reference"] == mode]
            modes.append(dict(run=name, mode=mode, n=len(part),
                successes=sum(x["success30"] for x in part), baseline_successes=sum(x["success30"] for x in old),
                **transitions(old, part)))
        evidence[name] = dict(run_config_sha256=digest(run/"run_config.json"),
            refinement_config_sha256=digest(run/"recovery_bc_config.json"),
            episodes_sha256=digest(folder/"episodes.json"), audit_sha256=digest(folder/"native_replay_audit.json"))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    paired_diagnostics = []
    forced = BASE/"D27_teacher_forcing_matched_inference"
    if (forced/"summary.json").exists():
        assert read(forced/"summary.json")["status"] == "completed"
        closed = BASE/"D22_student_fixed_recovery_state"
        for name in ("I27_phase_s13102", "B04_phase_recovery_data"):
            filename = f"{name}_seed90523.npz"
            with np.load(closed/filename) as a, np.load(forced/filename) as b:
                assert set(a.files) == set(b.files)
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
            paired_diagnostics.append(dict(run=name, all_trace_arrays_identical=True,
                closed_trace_sha256=digest(closed/filename), forced_trace_sha256=digest(forced/filename)))
        (output/"matched_inference_check.json").write_text(json.dumps(dict(
            phase_controls=paired_diagnostics, diagnostic_results=read(forced/"results.json"),
            scope="One seen training state; oracle recorded observations are not deployable"), indent=2)+"\n")
    for filename, records in (("overview.csv", rows), ("orientation.csv", modes)):
        with (output/filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (output/"evidence.json").write_text(json.dumps(dict(status="completed_development_only",
        baseline_run=str(control), baseline_step=0, baseline_episode_sha256=digest(baseline/"episodes.json"),
        baseline_checkpoint_sha256=baseline_cfg["checkpoint_sha256"], evidence=evidence,
        source_sha256=digest(Path(__file__))), indent=2)+"\n")
    lines = ["# 反馈BC与恢复示范：学生是否真的学会", "",
        "任务、物理、30秒严格成功门槛不变。全部是同64个开发初态、单个监督采样种子；不是独立测试或完整DAgger。",
        "原I27 BC在该初态集26/64=40.625%，取E27零步记录并核对源权重哈希；不是I27原来另一组初态的分数。",
        "每个新组从同I27开始追加4000次BC更新，PPO步数0。first/last/max只是同一次训练结束评估，不能冒充训练中成功率曲线。", "",
        "| 组 | 成功/64 | 成功率 | 相对原BC新增/丢失 | 平均合格秒 | 最大穿透mm | 验证动作损失 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['run']} | {row['successful_episodes']} | {row['success30']:.2%} | {row['gained']}/{row['lost']} | {row['hold_sec']:.3f} | {row['max_depth_mm']:.3f} | {row['validation_loss']:.6g} |")
    lines += ["", "## 读表方式", "",
        "B02开放反馈但只用原数据；B03增加1条经过物理验证的恢复示范；B04用同恢复数据但仍只看阶段/朝向；B05仅排除新增轨迹中学生生成的前3秒标签；B06对照B03仅屏蔽当前控制/上一动作输入。B07/B08使用D36的2条早期干预恢复示范，分别开放反馈/仅阶段输入。未列出的组不能据此推断已完成。",
        "全部失败均保留在分母和接触质量统计中。表中最大穿透包括失败案例，不代表成功案例被放宽门槛。保存的预览均有真实权重动作及原生物理审计，不冒充64条均另存重放。",
        "新增的是仿真优化的训练恢复示范，不是新视频，14条原验证轨迹没有加入训练。每组具体数量与来源见其dataset_manifest；教师额外搜索预算单列于教师报告，不计作免费示范。", "",
        "## 机制诊断", "",
        "D22从同一个可救的3秒状态接手，原BC、B02、B03、B04都不能成功。B03虽更接近教师动作，也只合格0.4秒。D23喂教师记录观测时B03形成32.55秒连续接触，但穿透4.072mm，因此依旧失败。它使用未来教师观测，不能部署或当新方法成绩。",
        "D27进一步统一两条路径的逐步推理形状，排除离线大批量GEMM舍入差异：I27/B04的全部21个轨迹数组与D22逐元素相同；B03依旧32.55秒连续接触但穿透4.115mm、严格失败。这支持继续调查状态分布偏离和不稳定反馈，但尚未证明具体哪个输入为根因。matched_inference_check.json保存该核查。" if paired_diagnostics else "D27同推理形状对照尚未汇总。",
        "这些结果支持继续检查反馈关系和状态分布偏离，不证明只靠多加BC损失即可解决。精确方案、论文与后续日志见项目docs/experiments/S09/论文启发的反馈BC与残差改进_2026-10-19.md。",
        "", "overview.csv为完整质量/得失统计，orientation.csv按朝向拆分，evidence.json绑定源文件。"]
    (output/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
