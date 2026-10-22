"""Compare the new expert branch to the same steps of its single-actor control.

Development diagnostics only. A short expert run is never compared against
the final checkpoint of a longer control run.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"
RUN = BASE/"E77_bc_late_experts"
CONTROL = BASE/"E46_bc_budget_continue_s14122"


def select_runs(long=False, replica=None, long_replica=None):
    if sum((bool(long), replica is not None, long_replica is not None)) > 1:
        raise ValueError("Choose one analysis protocol")
    seed = long_replica if long_replica is not None else replica
    if seed is not None and seed not in (14121, 14123):
        raise ValueError("Unknown training source")
    steps = 655360 if long or long_replica is not None else 163840
    if seed is not None:
        control = BASE/("E55_bc_budget_restart_s14121" if seed == 14121 else "E56_bc_budget_restart_s14123")
        if long_replica is not None:
            run = BASE/("E104_experts_long_s14121" if seed == 14121 else "E105_experts_long_s14123")
            return run, control, f"late_experts_long_s{seed}", steps
        run = BASE/("E79_bc_late_experts_s14121" if seed == 14121 else "E80_bc_late_experts_s14123")
        return run, control, f"late_experts_s{seed}", steps
    return (BASE/"E78_bc_late_experts_long" if long else RUN), CONTROL, (
        "late_experts_long" if long else "late_experts"), steps


def transitions(before, after):
    if not before or len(before) != len(after):
        raise ValueError("Nonempty, paired episodes are required")
    if [r["seed"] for r in before] != [r["seed"] for r in after]:
        raise ValueError("Episode seeds must match in order")
    for left, right in zip(before, after):
        if left["selected_reference"] != right["selected_reference"]:
            raise ValueError("Initial orientation routing changed")
    return dict(gained=sum(bool(b["success30"]) and not a["success30"] for a,b in zip(before,after)),
                lost=sum(bool(a["success30"]) and not b["success30"] for a,b in zip(before,after)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--long", action="store_true", help="Analyze E78's full 655360-step stage separately")
    choice.add_argument("--replica", type=int, choices=(14121,14123), help="Analyze an independent-source short repeat")
    choice.add_argument("--long-replica", type=int, choices=(14121,14123), help="Analyze a full 655360-step independent-source repeat")
    args = parser.parse_args()
    run, control, report_name, expected_steps = select_runs(args.long, args.replica, args.long_replica)
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    cfg, control_cfg = [read(p/"run_config.json") for p in (run, control)]
    for key in ("seed", "std", "learning_rate", "model_sha256", "reference_sha256",
                "method", "env_config", "eval_start", "eval_count", "eval_batch", "backend"):
        assert cfg[key] == control_cfg[key], key
    assert cfg["policy_architecture"] == "six initial-pose-routed full-target actors"
    for filename in cfg["sources"].keys() & control_cfg["sources"].keys():
        assert cfg["sources"][filename] == control_cfg["sources"][filename], filename
    assert cfg["steps"] == expected_steps and cfg["evals"] == expected_steps//81920+1
    assert cfg["eval_start"] == 130000 and cfg["eval_count"] == 64
    for filename in ("constraint_config.json", "improvement_config.json"):
        current, previous = [read(p/filename) for p in (run, control)]
        keys = ("actor_learning_rate", "critic_learning_rate", "teacher_kl_weight") if filename.startswith("constraint") else ("training_overrides", "initial_std_override")
        for key in keys:
            assert current[key] == previous[key], key
    derived = Path(cfg["checkpoint"])
    conversion = read(derived.parent/"conversion.json")
    assert conversion["source_checkpoint_sha256"] == control_cfg["checkpoint_sha256"]
    assert conversion["derived_checkpoint_sha256"] == cfg["checkpoint_sha256"] == sha(derived)
    assert conversion["all_six_actor_copies_bitwise_identical"]
    assert conversion["critic_and_normalizer_unchanged"]
    assert sha(Path(conversion["source_checkpoint"])) == control_cfg["checkpoint_sha256"]
    summary = read(run/"summary.json")
    assert summary["status"] == "completed" and summary["final_step"] == cfg["steps"]
    with (run/"metrics.csv").open() as stream:
        metrics = [{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
    with (control/"metrics.csv").open() as stream:
        control_metrics = {int(r["step"]):r for r in csv.DictReader(stream)}
    assert [int(row["step"]) for row in metrics] == list(range(0, expected_steps+1, 81920))
    clone_audit = read(run/"step_000000000/clone_behavior_audit.json")
    assert clone_audit["status"] == "verified" and clone_audit["parameters_exactly_cloned"]
    assert clone_audit["source_checkpoint_sha256"] == control_cfg["checkpoint_sha256"]
    assert clone_audit["clone_checkpoint_sha256"] == sha(run/"step_000000000/params.pkl")
    rows, modes, initial_differences = [], [], None
    first = read(run/"step_000000000/episodes.json")
    for row in metrics:
        step = int(row["step"])
        current, previous = [p/f"step_{step:09d}" for p in (run, control)]
        with np.load(current/"initial_states.npz") as a, np.load(previous/"initial_states.npz") as b:
            for key in ("seeds", "qpos"):
                np.testing.assert_array_equal(a[key], b[key])
        after, before = [read(p/"episodes.json") for p in (current, previous)]
        assert len(after) == 64
        if step == 0:
            initial_differences = {k:sum(a[k]!=b[k] for a,b in zip(before, after)) for k in before[0]}
        assert np.mean([r["success30"] for r in after]) == row["success30"]
        control_rate = np.mean([r["success30"] for r in before])
        assert control_rate == float(control_metrics[step]["success30"])
        rows.append(dict(step=step, expert_success=row["success30"], control_success=control_rate,
            expert_hold_sec=row["hold_sec"], control_hold_sec=float(control_metrics[step]["hold_sec"]),
            expert_max_depth_mm=1000*row["max_penetration_m"], expert_peak_force_n=row["peak_force_n"],
            **transitions(before, after)))
        for mode in range(6):
            part = [r for r in after if r["selected_reference"] == mode]
            old = [r for r in before if r["selected_reference"] == mode]
            modes.append(dict(step=step, mode=mode, n=len(part),
                expert_successes=sum(r["success30"] for r in part),
                control_successes=sum(r["success30"] for r in old)))
    folder = run/f"step_{cfg['steps']:09d}"
    audit = read(folder/"native_replay_audit.json")
    assert audit["status"] == "verified" and audit["normalized_action_tolerance"] == 1e-5
    assert audit["checkpoint_sha256"] == sha(folder/"params.pkl")
    assert audit["trace_sha256"] == sha(folder/"preview_trajectories.npz")
    best = max(rows, key=lambda r:r["expert_success"])
    report = dict(status="completed_development_only", training_repetitions=1,
        first=rows[0]["expert_success"], last=rows[-1]["expert_success"],
        max=best["expert_success"], best_step=best["step"],
        hold_summary=summary["hold_sec"], own_start_transitions=transitions(first, after),
        actor_parameter_multiplier=6, source_sha256=sha(Path(__file__)),
        run_config_sha256=sha(run/"run_config.json"), control_config_sha256=sha(control/"run_config.json"),
        conversion=conversion, rows=rows)
    report.update(clone_audit=clone_audit, initial_different_episode_fields=initial_differences)
    output = BASE/"report"/report_name
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (("curve.csv", rows), ("orientation.csv", modes)):
        with (output/name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (output/"statistics.json").write_text(json.dumps(report, indent=2)+"\n")
    title = "E78：独立分朝向专家的完整长阶段" if args.long else "E77：独立分朝向专家的后期接续探测"
    if args.replica or args.long_replica:
        title = f"{run.name}：独立来源重复，对照{control.name}"
    lines = ["# "+title, "",
        f"只使用既有64个开发初态。单次训练、actor容量六倍，不是同容量比较，也不是独立保留验证。控制是{control.name}在相同步数的检查点，不是其更晚末步。", "",
        "| 阶段追加交互 | 六专家 | 原单网络 | 新增成功 | 丢失成功 |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['step']} | {row['expert_success']:.2%} | {row['control_success']:.2%} | {row['gained']} | {row['lost']} |")
    lines += ["", f"专家success first={report['first']:.2%}，last={report['last']:.2%}，max={report['max']:.2%}，首次best_step={report['best_step']}。",
        f"相对自身阶段起点：{report['own_start_transitions']}。阶段起点此前已做655360次PPO交互，不是原始BC。",
        f"末步全部案例释放后最大穿透{rows[-1]['expert_max_depth_mm']:.3f}mm，峰值力{rows[-1]['expert_peak_force_n']:.3f}N，失败不筛除。",
        "", f"源actor参数逐元素复制、不再拟合数据；同输入动作误差{clone_audit['max_normalized_action_error']:.3g}，通过1e-5容差核验。GPU不同计算图的微小舍入会在接触仿真中累积，不能宣称初始物理轨迹逐元素相同。起点逐例不同字段计数：{initial_differences}。",
        "末步保存预览通过原1e-5容差的权重/原生物理回放。审计不冒充所有未保存轨迹均重放。",
        "", "六专家依旧输出完整关节目标，不是参考加残差；价值网络仍共享。未更换视频、奖励或物理任务，不把本组结果写回原确认表。",
        "", f"curve.csv含各时点接触质量，orientation.csv含各朝向计数，statistics.json含来源哈希和摘要；完整逐例数据在../../{run.name}各step目录。"]
    (output/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
