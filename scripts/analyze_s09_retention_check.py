"""Analyze all five frozen jobs; never publish a favorable partial subset."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018/confirmation_retention_20261019"
LABELS = {"bc": "原始BC", "stage_start": "本阶段起点", "control_last": "原配方真正末步",
          "control_selected": "原配方预选模型", "candidate": "保持约束真正末步"}


def check_source_hashes(runtime, frozen, historical):
    """Keep precommitted hashes separate from retrospective source evidence."""
    supplemental = {}
    for name, expected in runtime.items():
        if Path(name).name != name:
            raise ValueError("Expected a source basename")
        key = f"scripts/{name}"
        if key in frozen:
            if frozen[key] != expected:
                raise ValueError(f"Frozen source changed: {name}")
        else:
            if not historical or any(sources.get(name) != expected for sources in historical):
                raise ValueError(f"Shared source lacks matching historical evidence: {name}")
            supplemental[name] = expected
    return supplemental


def paired_result(candidate, control, seed=170001, draws=10000):
    candidate, control = np.asarray(candidate), np.asarray(control)
    if candidate.ndim != 1 or candidate.shape != control.shape or not len(candidate):
        raise ValueError("Use nonempty paired one-dimensional labels")
    if not np.isin(candidate, (0, 1)).all() or not np.isin(control, (0, 1)).all():
        raise ValueError("Success labels must be binary")
    candidate, control = candidate.astype(bool), control.astype(bool)
    delta = candidate.astype(float)-control.astype(float)
    indices = np.random.default_rng(seed).integers(0, len(delta), size=(draws, len(delta)))
    interval = np.quantile(delta[indices].mean(axis=1), [.025, .975])*100
    return dict(gained=int(np.sum(candidate & ~control)), lost=int(np.sum(~candidate & control)),
        both_success=int(np.sum(candidate & control)), both_failure=int(np.sum(~candidate & ~control)),
        gain_pp=float(delta.mean()*100), ci95_low_pp=float(interval[0]), ci95_high_pp=float(interval[1]))


def main():
    plan_path = BASE/"plan.json"
    plan = json.loads(plan_path.read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    plan_hash = digest(plan_path)
    for path, expected in plan["execution_sources"].items():
        assert digest(ROOT/path) == expected, path
    origins = [ROOT/job["checkpoint"] for job in plan["jobs"]]
    origin_configs = [path.parent.parent/"run_config.json" for path in origins]
    historical = [json.loads(path.read_text())["sources"] for path in origin_configs]
    supplemental = None
    rows, episodes, labels, scenes = [], [], {}, None
    orientation = []
    for job in plan["jobs"]:
        run = ROOT/job["output"]
        # Require both completed evaluation and unchanged-tolerance audit.
        binding = json.loads((run/"retention_check_binding.json").read_text())
        assert binding["completed_with_preview_audit"] and binding["plan_sha256"] == plan_hash
        assert binding["checkpoint_sha256"] == job["checkpoint_sha256"]
        assert digest(ROOT/job["checkpoint"]) == job["checkpoint_sha256"]
        summary = json.loads((run/"summary.json").read_text())
        assert summary["status"] == "completed" and summary["final_step"] == 0
        cfg = json.loads((run/"run_config.json").read_text())
        assert cfg["method"] == "guided" and cfg["steps"] == 0 and cfg["backend"] == "gpu"
        assert cfg["checkpoint_sha256"] == job["checkpoint_sha256"]
        for key in ("eval_start", "eval_count", "model_sha256", "reference_sha256"):
            assert cfg[key] == plan[key], (job["name"], key)
        extra = check_source_hashes(cfg["sources"], plan["execution_sources"], historical)
        if supplemental is None:
            supplemental = extra
        assert extra == supplemental
        for name, expected in cfg["sources"].items():
            assert digest(ROOT/"scripts"/name) == expected, name
            assert digest(run/"sources"/name) == expected, name
        # These common dependencies were saved by all training runs but omitted
        # from the frozen plan. Audit them explicitly; do not rewrite the plan.
        for name, expected in extra.items():
            for origin in origins:
                assert digest(origin.parent.parent/"sources"/name) == expected, name
        folder = run/"step_000000000"
        audit = json.loads((folder/"native_replay_audit.json").read_text())
        assert audit["status"] == "verified" and audit["normalized_action_tolerance"] == 1e-5
        assert audit["trace_sha256"] == digest(folder/"preview_trajectories.npz")
        assert audit["checkpoint_sha256"] == digest(folder/"params.pkl")
        with np.load(folder/"initial_states.npz") as initial:
            np.testing.assert_array_equal(initial["seeds"], np.arange(plan["eval_start"], plan["eval_start"]+plan["eval_count"]))
            if scenes is None:
                scenes = initial["qpos"].copy()
            np.testing.assert_array_equal(initial["qpos"], scenes)
        case = json.loads((folder/"episodes.json").read_text())
        assert [r["seed"] for r in case] == list(range(plan["eval_start"], plan["eval_start"]+plan["eval_count"]))
        success = np.array([r["success30"] for r in case], bool)
        assert success.mean() == summary["rows"][0]["success30"]
        labels[job["name"]] = success
        episodes.extend(dict(group=job["name"], **r) for r in case)
        rows.append(dict(group=job["name"], n=len(case), successes=int(success.sum()), success30=float(success.mean()),
            hold_sec=float(np.mean([r["hold_sec"] for r in case])),
            secondary_all_depth_success=float(np.mean([r["success30_all_depth_le_3mm"] for r in case])),
            max_released_depth_mm=max(r["max_released_penetration_m"] for r in case)*1000,
            max_pre_release_depth_mm=max(r["max_pre_release_penetration_m"] for r in case)*1000,
            max_peak_force_n=max(r["peak_force_n"] for r in case),
            gate_disagreement_frames=sum(r["failure_gates"]["valid_reconstruction_disagreement_frames"] for r in case),
            audited_preview_episodes=len(audit["episodes"])))
        for mode in range(6):
            part = [r for r in case if r["selected_reference"] == mode]
            orientation.append(dict(group=job["name"], mode=mode, n=len(part), successes=sum(r["success30"] for r in part)))
    paired = [dict(control=job["name"], **paired_result(labels["candidate"], labels[job["name"]]))
              for job in plan["jobs"] if job["name"] != "candidate"]
    output = BASE/"report"
    output.mkdir(exist_ok=True)
    for filename, values in (("episodes.csv", episodes), ("paired.csv", paired), ("orientation.csv", orientation)):
        with (output/filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows({k:json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k,v in row.items()} for row in values)
    result = dict(plan_sha256=plan_hash, training_repetitions=1, cases=plan["eval_count"],
        uncertainty="10000 paired bootstrap draws across initial states only; no training-seed uncertainty; four intervals unadjusted for multiplicity",
        supplemental_source_audit=dict(scope="Retrospective, not precommitted in plan; unchanged across all five historical training snapshots, evaluation snapshots and current files",
            shared_sources=supplemental,
            historical_configs={str(path.relative_to(ROOT)):digest(path) for path in origin_configs}),
        source_sha256=digest(Path(__file__)), rows=rows, paired=paired)
    (output/"statistics.json").write_text(json.dumps(result, indent=2)+"\n")
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    x = np.arange(len(rows))
    ax.bar(x, [100*r["success30"] for r in rows], color=["#777777", "#a5a5a5", "#777777", "#ba842e", "#007f73"])
    ax.set_xticks(x, ["Original BC", "Stage start", "Original last", "Original selected", "Retention last"])
    ax.set(ylim=(0, 100), ylabel="30-second success (%)", title="Fresh initial states; ONE training repetition")
    for i, row in enumerate(rows):
        ax.text(i, 100*row["success30"]+2, f"{row['successes']}/{row['n']}", ha="center")
    fig.savefig(output/"success.png", dpi=160)
    plt.close(fig)
    lines = ["# 长阶段保持约束：固定候选后的新初态检查", "",
        "五组均在160000--160255同256初态评估，全部1280次执行保留，权重在使用这些初态之前已冻结。本轮只有一个训练重复，不是五次独立训练，也不能替代三种子稳定性验证。", "",
        "| 策略 | 成功/256 | 成功率 | 最长合格时长均值(s) |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {LABELS[row['group']]} | {row['successes']}/{row['n']} | {row['success30']:.2%} | {row['hold_sec']:.3f} |")
    lines += ["", "![新初态结果](success.png)", "", "## 相同初态上的新增与丢失", "",
        "以下始终是保持约束末步减去对应对照，不能根据测试结果再换本轮检查点。", "",
        "| 对照 | 新增成功 | 丢失成功 | 差值(pp) | 描述性95%区间(pp) |", "|---|---:|---:|---:|---:|"]
    for row in paired:
        lines.append(f"| {LABELS[row['control']]} | {row['gained']} | {row['lost']} | {row['gain_pp']:+.2f} | [{row['ci95_low_pp']:+.2f}, {row['ci95_high_pp']:+.2f}] |")
    lines += ["", "区间仅重采样初态，未覆盖训练随机性，四项比较没有多重校正。经验区间退化成单点也不表示真实差值已确定；不作算法普遍优越性宣称。", "",
        "## 接触与审核", "", "| 策略 | 全部案例释放后最大穿透(mm) | 释放前最大穿透(mm) | 峰值力(N) | 全回合3mm次级成功率 |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {LABELS[row['group']]} | {row['max_released_depth_mm']:.3f} | {row['max_pre_release_depth_mm']:.3f} | {row['max_peak_force_n']:.3f} | {row['secondary_all_depth_success']:.2%} |")
    lines += ["", f"全部案例门控重建差异帧合计{sum(r['gate_disagreement_frames'] for r in rows)}；{sum(r['audited_preview_episodes'] for r in rows)}条保存预览通过权重绑定与独立物理回放，不冒充所有1280条轨迹均独立重放。失败的高穿透和高力不删除；仿真成功不是硬件安全证明。", "",
        f"源码记录限制：冻结清单包含96项，但未单列{len(supplemental)}项共用依赖。汇总时另核对了这些依赖在五组原训练配置、原训练源码快照、此次评估源码快照与当前文件中的哈希，全部一致；补充证据写入statistics.json。此项是事后完整性核对，不冒充事先冻结全部依赖，未改原计划、评分或任何评估轨迹。", "",
        "原BC尚未在线强化学习；阶段起点已经累计655360次交互；两种末步路径总量均1310720。原配方预选模型是旧开发阶段已固定的E46/327680，不是本次看测试结果新挑选，所选权重路径983040但原实验完整训练预算仍1310720。相同在线预算不等于全部搜索成本相同。", "",
        "所有策略都是同一17维完整目标网络。保持约束的教师只参与训练，不是执行时BC动作加PPO修正，也不是残差架构。本轮没有更换视频、手型、摩擦或成功条件。", "",
        "逐例数据见episodes.csv，配对得失见paired.csv，各朝向见orientation.csv，固定权重和来源见../plan.json。E76类别均衡BC接PPO是另一个开发分支，未插入本次五组。"]
    (output/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
