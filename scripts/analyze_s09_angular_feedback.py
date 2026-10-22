"""Audit and pair the angular-input intervention with its fixed PPO control."""

import csv
import json
from pathlib import Path

import numpy as np

from analyze_s09_late_experts import BASE, transitions
from retarget_s09_reference import digest


def paired_initials(left, right):
    for key in ("seeds", "qpos"):
        np.testing.assert_array_equal(left[key], right[key])


def main():
    run = BASE/"E81_bc_angular_velocity"
    control = BASE/"E27_bc_fast_critic_s14102"
    read = lambda path: json.loads(path.read_text())
    cfg, old = [read(p/"run_config.json") for p in (run, control)]
    for key in ("seed", "std", "learning_rate", "model_sha256", "reference_sha256",
                "method", "eval_start", "eval_count", "eval_batch", "backend"):
        assert cfg[key] == old[key], key
    assert cfg["steps"] == 40960 and cfg["evals"] == 2
    assert cfg["eval_start"] == 130000 and cfg["eval_count"] == 64
    env_config = dict(cfg["env_config"])
    feature = env_config.pop("angular_velocity_observation")
    assert feature == "mujoco_freejoint_angular_qvel_x0.1_clip5_slots101_104"
    assert env_config == old["env_config"]
    for filename in cfg["sources"].keys() & old["sources"].keys():
        assert cfg["sources"][filename] == old["sources"][filename], filename
    for filename, keys in (
        ("constraint_config.json", ("actor_learning_rate", "critic_learning_rate", "teacher_kl_weight")),
        ("improvement_config.json", ("training_overrides", "initial_std_override")),
    ):
        a, b = [read(p/filename) for p in (run, control)]
        for key in keys:
            assert a[key] == b[key], key
    conversion_path = Path(cfg["checkpoint"]).parent/"conversion.json"
    conversion = read(conversion_path)
    assert conversion["source_checkpoint_sha256"] == old["checkpoint_sha256"]
    assert conversion["derived_checkpoint_sha256"] == cfg["checkpoint_sha256"] == digest(Path(cfg["checkpoint"]))
    summary = read(run/"summary.json")
    assert summary["status"] == "completed" and summary["final_step"] == 40960
    metrics = summary["rows"]
    assert [r["step"] for r in metrics] == [0, 40960]
    old_metrics = {r["step"]: r for r in read(control/"summary.json")["rows"]}
    rows, modes, evidence = [], [], {}
    for step in (0, 40960):
        folder, baseline = [p/f"step_{step:09d}" for p in (run, control)]
        with np.load(folder/"initial_states.npz") as a, np.load(baseline/"initial_states.npz") as b:
            paired_initials(a, b)
        current, previous = [read(p/"episodes.json") for p in (folder, baseline)]
        assert len(current) == 64
        audit = read(folder/"native_replay_audit.json")
        assert audit["status"] == "verified" and audit["feature"] == feature
        assert audit["checkpoint_sha256"] == digest(folder/"params.pkl")
        assert audit["trace_sha256"] == digest(folder/"preview_trajectories.npz")
        assert audit["normalized_action_tolerance"] == 1e-5
        metric = next(r for r in metrics if r["step"] == step)
        assert np.mean([r["success30"] for r in current]) == metric["success30"]
        assert np.mean([r["success30"] for r in previous]) == old_metrics[step]["success30"]
        delta = transitions(previous, current)
        if step == 0:
            first = current
            initial_delta = delta
            initial_field_differences = {k: sum(a[k] != b[k] for a, b in zip(current, previous))
                                         for k in current[0]}
        rows.append(dict(step=step, angular_success=metric["success30"],
            control_success=old_metrics[step]["success30"], angular_hold_sec=metric["hold_sec"],
            control_hold_sec=old_metrics[step]["hold_sec"],
            max_released_depth_mm=1000*metric["max_penetration_m"],
            peak_force_n=metric["peak_force_n"], forbidden_rate=metric["forbidden_rate"],
            released_rope_force_n=metric["released_rope_force_n"], **delta))
        for mode in range(6):
            a = [r for r in current if r["selected_reference"] == mode]
            b = [r for r in previous if r["selected_reference"] == mode]
            modes.append(dict(step=step, mode=mode, n=len(a),
                angular_successes=sum(r["success30"] for r in a),
                control_successes=sum(r["success30"] for r in b), **transitions(b, a)))
        evidence[str(step)] = dict(episodes_sha256=digest(folder/"episodes.json"),
            control_episodes_sha256=digest(baseline/"episodes.json"),
            audit_sha256=digest(folder/"native_replay_audit.json"))
    result = dict(status="completed_development_only", success=summary["success30"],
        hold=summary["hold_sec"], own_start_transitions=transitions(first, current),
        initial_paired_changes=initial_delta, initial_field_differences=initial_field_differences,
        conversion=conversion, config_sha256=digest(run/"run_config.json"),
        control_config_sha256=digest(control/"run_config.json"), source_sha256=digest(Path(__file__)),
        evidence=evidence, rows=rows,
        limitations="64 reused development states, one training seed; different total training budget/eval schedule from the historical control, compare its same-step checkpoint, not a bitwise training replication")
    output = BASE/"report/angular_feedback"
    output.mkdir(parents=True, exist_ok=True)
    (output/"statistics.json").write_text(json.dumps(result, indent=2)+"\n")
    for name, values in (("curve.csv", rows), ("orientation.csv", modes)):
        with (output/name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    lines = ["# E81: PPO增加圆柱角速度输入", "",
        "同一I27 BC起点，完整动作网络，不是残差。新增输入前将原恒零输入对应权重清零，保持初始函数；不改奖励、物理和成功判据。", "",
        "| 步数 | 角速度组 | 同步数旧控制 | 相对控制新增/丢失 | 平均合格秒 | 最大穿透mm | 峰值N |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['step']} | {r['angular_success']:.2%} | {r['control_success']:.2%} | {r['gained']}/{r['lost']} | {r['angular_hold_sec']:.3f} | {r['max_released_depth_mm']:.3f} | {r['peak_force_n']:.3f} |")
    lines += ["", f"success first/last/max/best_step: {summary['success30']}",
        f"相对自身BC起点新增/丢失: {result['own_start_transitions']}",
        f"零步成功标签变化: {initial_delta}；全部字段差异计数见statistics.json。",
        "", "两次保存预览通过权重动作、真实前一状态角速度、原生物理回放核验；不冒充所有64例另存重放。最大穿透包括失败案例。",
        "64个反复使用的开发初态、单种子，不能作为独立泛化结论。历史控制E27总预算163840、5次评估，本组40960、2次评估；比较的是相同40960检查点，不宣称训练随机路径逐位相同。",
        "此实验没有新增恢复示范，没有证明BC已经学会纠错。权重需用角速度输入入口加载，旧环境填零会改变执行策略。"]
    (output/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
