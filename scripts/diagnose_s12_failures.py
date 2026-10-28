"""Separate drops and post-release quality violations using saved S12 traces."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_s12_study import ORDER, csv_write, plt


def diagnose(trace, config):
    t = np.asarray(trace["time"])
    released = t >= config["release_sec"] + .5 - 1e-8
    gates = {
        "strain": trace["max_principal_strain"] > config["max_strain"],
        "penetration": trace["penetration_m"] > config["max_penetration"],
        "force": trace["peak_force_n"] > config["max_force"],
        "forbidden_contact": trace["forbidden_force_n"] >= .02,
    }
    masks = {name: np.asarray(mask) & released for name, mask in gates.items()}
    bad = np.logical_or.reduce(list(masks.values()))
    alive = np.asarray(trace["alive"], bool)
    collapsed = np.asarray(trace["min_volume_ratio"]) <= .05
    inverted = np.asarray(trace["min_volume_ratio"]) <= 0
    dropped = np.asarray(trace["vertices"]).mean(axis=1)[:, 2] <= .4
    strict = bool(trace["success15"][-1])
    if strict and (bad.any() or not alive[-1] or collapsed.any()):
        raise ValueError("Recorded success contradicts post-release quality/drop trace")
    reason = ("passed" if strict else "element_collapse_or_inversion" if collapsed.any() else
              "dropped" if dropped.any() else
              "retained_with_quality_violation" if bad.any() else
              "retained_without_required_strict_hold")
    row = dict(failure_class=reason, final_alive=bool(alive[-1]),
        dropped=bool(dropped.any()), element_collapse=bool(collapsed.any()),
        element_inversion=bool(inverted.any()),
        first_element_collapse_sec=float(t[np.flatnonzero(collapsed)[0]]) if collapsed.any() else None,
        final_strict_hold_sec=float(trace["hold_sec"][-1]),
        first_drop_sec=float(t[np.flatnonzero(dropped)[0]]) if dropped.any() else None,
        first_quality_violation_sec=float(t[np.flatnonzero(bad)[0]]) if bad.any() else None,
        post_release_forbidden_max_n=float(np.max(trace["forbidden_force_n"][released])) if released.any() else None,
        post_release_strain_max=float(np.max(trace["max_principal_strain"][released])) if released.any() else None)
    for name, mask in masks.items():
        row[name + "_violating_control_intervals"] = int(mask.sum())
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads((a.study / "evaluation_plan.json").read_text())
    rows, groups = [], []
    for job in plan["jobs"]:
        run = Path(job["output"])
        summary = json.loads((run / "summary.json").read_text())
        episodes = json.loads((run / "episodes.json").read_text())
        cfg = json.loads((run / "evaluation_config.json").read_text())["soft_config"]
        if summary["status"] != "completed" or len(episodes) != job["eval_count"]:
            raise ValueError("Cannot diagnose an incomplete evaluation: " + job["name"])
        for episode in episodes:
            path = run / f"trajectory_{episode['seed']}.npz"
            with np.load(path) as trace:
                if bool(trace["success15"][-1]) != episode["success15"]:
                    raise ValueError("Trace/record mismatch")
                row = dict(method=job["label"], train_seed=job["train_seed"],
                    condition=job["condition"], initial_seed=episode["seed"],
                    strict_success=episode["success15"], physical_success=episode["physical_success15"],
                    **diagnose(trace, cfg), trace=str(path))
                rows.append(row)
    for condition, method in dict.fromkeys((r["condition"], r["method"]) for r in rows):
        group = [r for r in rows if r["condition"] == condition and r["method"] == method]
        groups.append(dict(condition=condition, method=method, episodes=len(group),
            strict_successes=sum(r["strict_success"] for r in group),
            physical_successes=sum(r["physical_success"] for r in group),
            dropped=sum(r["dropped"] for r in group),
            element_collapse=sum(r["element_collapse"] for r in group),
            element_inversion=sum(r["element_inversion"] for r in group),
            quality_violation=sum(r["first_quality_violation_sec"] is not None for r in group),
            retained_with_quality_violation=sum(r["failure_class"] == "retained_with_quality_violation" for r in group),
            retained_without_required_strict_hold=sum(r["failure_class"] == "retained_without_required_strict_hold" for r in group)))
    output = a.study / "report"
    output.mkdir(exist_ok=True)
    csv_write(output / "failure_episodes.csv", rows)
    csv_write(output / "failure_counts.csv", groups)
    categories = [("passed", "Strict pass", "#16816a"),
        ("element_collapse_or_inversion", "Invalid element", "#b64b5d"),
        ("dropped", "Drop", "#777777"),
        ("retained_with_quality_violation", "Quality violation", "#b57d22"),
        ("retained_without_required_strict_hold", "Insufficient strict hold", "#3b82b6")]
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    fig, axes = plt.subplots(1, len(conditions), figsize=(17, 4.8), sharey=True, squeeze=False)
    for ax, condition in zip(axes.flat, conditions):
        bottom = np.zeros(len(ORDER))
        totals = [sum(r["condition"] == condition and r["method"] == method for r in rows)
                  for method in ORDER]
        for category, label, color in categories:
            counts = np.array([sum(r["condition"] == condition and r["method"] == method
                and r["failure_class"] == category for r in rows) for method in ORDER])
            rates = 100 * counts / np.asarray(totals)
            ax.bar(np.arange(len(ORDER)), rates, bottom=bottom, label=label, color=color)
            bottom += rates
        np.testing.assert_allclose(bottom, 100.)
        ax.set_xticks(range(len(ORDER)), [f"{label}\nn={n}" for label,n in zip(
            ("PPO", "BC", "BC+PPO", "Reference", "Residual"), totals)], rotation=45, ha="right")
        ax.set(title=condition, ylim=(0, 100))
    axes[0,0].set_ylabel("Episodes (%)")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=9, bbox_to_anchor=(.5,.96))
    fig.suptitle("Mutually exclusive outcomes; pooled descriptive counts, not independent training trials", y=1)
    fig.tight_layout(rect=(0,0,1,.84))
    fig.savefig(output / "failure_types.png", dpi=160)
    plt.close(fig)
    lines = ["# S12 失败原因分解", "", "来自完整保存轨迹，不重新运行或改变成功门槛。",
        "违规按撤台完成后（5.5 秒起）计算；整个回合的峰值可能发生在撤台前，不能直接据此判定严格失败。",
        "违规持续数量是含至少一个违规物理子步的 50 ms 控制区间数，不等于接触精确持续时长。",
        "掉落、单元塌陷和质量违规可以同时发生，各列不可直接相加。体积比小于等于 0.05 会终止回合；非正值表示单元翻转，属于无效材料/数值状态，不能解释为真实物体已被捏碎。原始 alive=false 同时包含掉落与塌陷，不能只据它统计掉落。",
        "每种方法的多个训练种子这里只作描述性计数，不当成独立训练样本做显著性检验。",
        "", "| 条件 | 方法 | 回合数 | 严格通过 | 物理保持 | 掉落 | 单元塌陷 | 其中翻转 | 曾质量违规 | 未掉落但质量违规 | 未掉落但保持不足 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in groups:
        lines.append("| " + " | ".join(str(r[k]) for k in (
            "condition", "method", "episodes", "strict_successes", "physical_successes", "dropped",
            "element_collapse", "element_inversion", "quality_violation", "retained_with_quality_violation", "retained_without_required_strict_hold")) + " |")
    lines += ["", "![失败类型分布](failure_types.png)", "",
        "图中使用互斥的主分类：通过，其次局部单元失效，再其次掉落、未掉落但质量违规、其它保持不足。原表各个事件标志允许重叠；图的颜色加起来为 100%。多个网络的回合在图中只作描述性汇总，不当作同等数量的独立训练重复。",
        "逐回合时间点及每类门槛违反数量见 `failure_episodes.csv`。该表帮助定位现象，不单凭相关性确定训练退化的唯一因果。"]
    (output / "failure_analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(episodes=len(rows), report=str(output / "failure_analysis.md"))))


if __name__ == "__main__":
    main()
