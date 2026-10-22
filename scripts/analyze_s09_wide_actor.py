"""Paired capacity control: widened actor, routed actors, original actor."""

import csv
import json
from pathlib import Path
import pickle

import numpy as np

from analyze_s09_late_experts import BASE, transitions
from analyze_s09_angular_feedback import paired_initials
from retarget_s09_reference import digest


def capacity_updates(initial, final, old_width=128):
    rows = {}
    for key in ("hidden_0", "hidden_1", "hidden_2"):
        a, b = [np.asarray(p["params"][key]["kernel"]) for p in (initial, final)]
        if key == "hidden_0":
            a, b = a[:,old_width:], b[:,old_width:]
        else:
            a, b = a[old_width:], b[old_width:]
        rows[key] = dict(changed=int(np.count_nonzero(a != b)), l2=float(np.linalg.norm(b-a)))
    return rows


def main():
    read = lambda p: json.loads(p.read_text())
    runs = dict(wide=BASE/"E82_bc_wide_actor", experts=BASE/"E77_bc_late_experts",
                single=BASE/"E46_bc_budget_continue_s14122")
    configs = {k:read(p/"run_config.json") for k,p in runs.items()}
    cfg = configs["wide"]
    assert cfg["steps"] == 163840 and cfg["evals"] == 3
    assert cfg["policy_architecture"] == "single dense full-target actor 397x397 zero-output expansion"
    for name in ("experts", "single"):
        other = configs[name]
        for key in ("seed", "std", "learning_rate", "env_config", "model_sha256", "reference_sha256",
                    "eval_start", "eval_count", "eval_batch", "method", "backend"):
            assert cfg[key] == other[key], (name, key)
        for file in cfg["sources"].keys() & other["sources"].keys():
            assert cfg["sources"][file] == other["sources"][file], file
        a, b = [read(runs[n]/"constraint_config.json") for n in ("wide", name)]
        for key in ("actor_learning_rate", "critic_learning_rate", "teacher_kl_weight"):
            assert a[key] == b[key], key
        a, b = [read(runs[n]/"improvement_config.json") for n in ("wide", name)]
        for key in ("training_overrides", "initial_std_override"):
            assert a[key] == b[key], key
    conversion = read(Path(cfg["checkpoint"]).parent/"conversion.json")
    expert_conversion = read(Path(configs["experts"]["checkpoint"]).parent/"conversion.json")
    assert conversion["source_checkpoint_sha256"] == expert_conversion["source_checkpoint_sha256"] == configs["single"]["checkpoint_sha256"]
    assert digest(Path(cfg["checkpoint"])) == cfg["checkpoint_sha256"] == conversion["derived_checkpoint_sha256"]
    assert conversion["critic_and_normalizer_unchanged"]
    summary = read(runs["wide"]/"summary.json")
    assert summary["status"] == "completed" and summary["final_step"] == 163840
    metrics = {k:{r["step"]:r for r in read(p/"summary.json")["rows"]} for k,p in runs.items()}
    assert list(metrics["wide"]) == [0,81920,163840]
    rows, paired, modes, all_episodes, evidence = [], [], [], [], {}
    initial_fields = {}
    for step in (0,81920,163840):
        episodes = {}
        initial = None
        for name, run in runs.items():
            folder = run/f"step_{step:09d}"
            with np.load(folder/"initial_states.npz") as z:
                if initial is None:
                    initial = dict(z)
                paired_initials(initial, z)
            eps = read(folder/"episodes.json")
            assert len(eps) == 64
            assert np.mean([r["success30"] for r in eps]) == metrics[name][step]["success30"]
            if name == "wide":
                audit = read(folder/"native_replay_audit.json")
                assert audit["status"] == "verified" and audit["normalized_action_tolerance"] == 1e-5
                assert audit["checkpoint_sha256"] == digest(folder/"params.pkl")
                assert audit["trace_sha256"] == digest(folder/"preview_trajectories.npz")
            metric = metrics[name][step]
            rows.append(dict(group=name, step=step, success30=metric["success30"],
                hold_sec=metric["hold_sec"], max_depth_mm=1000*metric["max_penetration_m"],
                peak_force_n=metric["peak_force_n"], forbidden_rate=metric["forbidden_rate"],
                released_rope_force_n=metric["released_rope_force_n"]))
            for mode in range(6):
                part = [r for r in eps if r["selected_reference"] == mode]
                modes.append(dict(group=name, step=step, mode=mode, n=len(part),
                    successes=sum(r["success30"] for r in part)))
            all_episodes.extend(dict(group=name, step=step, **r) for r in eps)
            episodes[name] = eps
            evidence[f"{name}_{step}"] = dict(episodes_sha256=digest(folder/"episodes.json"),
                checkpoint_sha256=digest(folder/"params.pkl"))
        if step == 0:
            first = episodes["wide"]
            for name in ("single", "experts"):
                initial_fields[name] = {k:sum(a[k] != b[k] for a,b in zip(episodes["wide"], episodes[name]))
                                       for k in first[0]}
        for name in ("single", "experts"):
            paired.append(dict(step=step, contrast="wide minus "+name,
                difference=metrics["wide"][step]["success30"]-metrics[name][step]["success30"],
                **transitions(episodes[name], episodes["wide"])))
    params = [pickle.loads((runs["wide"]/f"step_{step:09d}/params.pkl").read_bytes())[1]
              for step in (0,163840)]
    result = dict(status="completed_development_capacity_control", success=summary["success30"],
        hold=summary["hold_sec"], own_start_transitions=transitions(first, episodes["wide"]),
        initial_field_differences=initial_fields, added_capacity_updates=capacity_updates(*params),
        conversion=conversion, evidence=evidence, rows=rows, paired=paired,
        source_sha256=digest(Path(__file__)), limitations="One origin, reused development initials; near-equal actor parameters, different architecture/initialization and no guarantee of equal effective capacity or compute")
    output = BASE/"report/wide_actor"
    output.mkdir(parents=True, exist_ok=True)
    (output/"statistics.json").write_text(json.dumps(result, indent=2)+"\n")
    for name, values in (("curve.csv", rows), ("paired.csv", paired), ("orientation.csv", modes), ("episodes.csv", all_episodes)):
        with (output/name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    lines = ["# E82：近六专家容量的单网络对照", "",
        "同E30阶段起点，累计前序655360步；三组仅比较追加0/81920/163840步。单网络36770、六专家220620、扩宽单网络220766个actor参数。", "",
        "| 组 | 追加步 | 成功率 | 平均合格秒 | 最大穿透mm | 峰值N |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['group']} | {r['step']} | {r['success30']:.2%} | {r['hold_sec']:.3f} | {r['max_depth_mm']:.3f} | {r['peak_force_n']:.3f} |")
    lines += ["", f"扩宽success摘要：{summary['success30']}", f"扩宽hold摘要：{summary['hold_sec']}",
        f"相对自身起点：{result['own_start_transitions']}", "", "| 比较 | 步 | 成功率差 | 新增/丢失 |", "|---|---:|---:|---:|"]
    for r in paired:
        lines.append(f"| {r['contrast']} | {r['step']} | {r['difference']:.2%} | {r['gained']}/{r['lost']} |")
    lines += ["", "不是独立保留评估，也不是算法普适排名。初始字段差异和新增参数实际更新量见statistics.json。",
        "三次扩宽保存预览均绑定真实权重并通过原生物理审计，动作容差不变；没有宣称全部64例均另存物理重放。失败案例计入穿透和峰值最大值。",
        "六专家训练前后仍存在不同初始化和参数共享方式，单次容量对照不能唯一确定因果。该组无角速度扩展、无新示范、无残差相加。"]
    (output/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
