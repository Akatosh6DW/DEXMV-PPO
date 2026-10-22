"""Development-only curves for the matched late-stage retention experiment."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"
RUNS = [("E46_bc_budget_continue_s14122", "Original continuation", "#696969"),
        ("E72_bc_stage_retention", "Selective retention", "#007f73")]


def read_curve(run):
    cfg = json.loads((run/"run_config.json").read_text())
    if (cfg["eval_start"], cfg["eval_count"]) != (130000, 64):
        raise ValueError("These plots are only for the declared development set")
    with (run/"metrics.csv").open() as stream:
        metrics = [{key:float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    if not metrics:
        raise ValueError("No completed evaluations yet")
    rows = []
    for record in metrics:
        step = int(record["step"])
        folder = run/f"step_{step:09d}"
        episodes = json.loads((folder/"episodes.json").read_text())
        if [row["seed"] for row in episodes] != list(range(130000, 130064)):
            raise ValueError("Unpaired development seeds")
        counts = np.bincount([row["selected_reference"] for row in episodes], minlength=6)
        success = np.bincount([row["selected_reference"] for row in episodes if row["success30"]], minlength=6)
        if success.sum()/64 != record["success30"] or (counts == 0).any():
            raise ValueError("Episode totals disagree with metrics")
        rows.append(dict(step=step, success=record["success30"], counts=counts, per_mode=success/counts,
            max_depth_mm=record["max_penetration_m"]*1000, episodes=episodes))
    complete = (run/"summary.json").exists()
    if complete and rows[-1]["step"] != cfg["steps"]:
        raise ValueError("Completed run does not reach its declared endpoint")
    return cfg, rows, complete


def main():
    series = [(name, label, color, *read_curve(BASE/name)) for name, label, color in RUNS]
    original = series[0][3]
    for _, _, _, cfg, _, _ in series[1:]:
        for key in ("checkpoint_sha256", "model_sha256", "reference_sha256", "seed", "steps", "evals",
                    "env_config", "learning_rate", "std", "method"):
            assert cfg[key] == original[key], key
    constraints = [json.loads((BASE/name/"constraint_config.json").read_text()) for name, *_ in series]
    assert constraints[0]["critic_learning_rate"] == constraints[1]["critic_learning_rate"]
    assert series[0][4][0]["episodes"] == series[1][4][0]["episodes"]
    bc = json.loads((BASE/"E27_bc_fast_critic_s14102/step_000000000/episodes.json").read_text())
    counts = np.bincount([row["selected_reference"] for row in bc], minlength=6)
    bc_success = np.bincount([row["selected_reference"] for row in bc if row["success30"]], minlength=6)
    output = BASE/"report"
    output.mkdir(exist_ok=True)
    exported = []
    for name, _, _, _, rows, complete in series:
        for row in rows:
            for mode in range(6):
                exported.append(dict(run=name, status="completed" if complete else "partial", step=row["step"],
                    mode=mode, mode_count=int(row["counts"][mode]), mode_success=float(row["per_mode"][mode]),
                    overall_success=row["success"], max_all_case_released_depth_mm=row["max_depth_mm"]))
    with (output/"stage_retention_curves.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for _, label, color, _, rows, complete in series:
        label += " (partial)" if not complete else ""
        steps = [r["step"]/1000 for r in rows]
        axes[0].plot(steps, [r["success"] for r in rows], marker="o", color=color, label=label)
        axes[1].plot(steps, [r["max_depth_mm"] for r in rows], marker="o", color=color, label=label)
    axes[0].axhline(bc_success.sum()/64, color="#b06400", linestyle=":", label="Original BC")
    axes[0].set(ylim=(0, 1), ylabel="30-second success", title="Same 64 development initial states")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set(ylim=(0, None), ylabel="Maximum released penetration (mm)", title="All cases, including failures")
    axes[1].axhline(3, color="#b5413c", linestyle=":", label="3 mm primary success gate")
    for axis in axes:
        axis.set(xlabel="Additional PPO interactions in this stage (thousands)", xlim=(0, 655.36))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, loc="upper right")
    fig.suptitle("Late-stage retention: development diagnostics, not held-out evidence", fontsize=11)
    fig.savefig(output/"stage_retention_overall.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), layout="constrained")
    for mode, axis in enumerate(axes.flat):
        for _, label, color, _, rows, complete in series:
            axis.plot([r["step"]/1000 for r in rows], [r["per_mode"][mode] for r in rows],
                      marker="o", color=color, label=label+(" (partial)" if not complete else ""))
        axis.axhline(bc_success[mode]/counts[mode], color="#b06400", linestyle=":", label="Original BC")
        axis.set(title=f"Pose bin {mode}, n={counts[mode]}"+(" (protected)" if mode in (0, 3, 5) else ""),
                 ylim=(0, 1.03), xlim=(0, 655.36), xlabel="Stage interactions (thousands)")
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("Initial-pose bins within uniform 3D orientations; same cases at every checkpoint", fontsize=11)
    fig.savefig(output/"stage_retention_modes.png", dpi=160)
    plt.close(fig)

    current, control = series[1][4], series[0][4]
    complete = series[1][5]
    first, last = current[0], current[-1]
    best = max(current, key=lambda r:r["success"])
    lines = ["# 长阶段保持约束：是否保住旧抓法并学到新抓法", "",
        "仅130000--130063这64个反复使用的开发初态，不是独立测试；当前只有一个训练重复。", "",
        "| 对象 | 成功/64 | 说明 |", "|---|---:|---|"]
    comparisons = [("原始I27 BC", bc, "尚未开始本轮PPO"),
        ("本阶段起点E30", first["episodes"], "前面已累计655360次交互，不是原BC"),
        ("无约束E46真正末步", control[-1]["episodes"], "本阶段再追加655360次交互")]
    if complete:
        comparisons.append(("保持约束E72真正末步", last["episodes"], "同样追加655360次交互"))
    else:
        comparisons.append(("保持约束E72部分结果", last["episodes"], f"仅完成到{last['step']}，不能与控制末步作等预算比较"))
    for name, episodes, note in comparisons:
        lines.append(f"| {name} | {sum(e['success30'] for e in episodes)}/64 | {note} |")
    lines += ["", f"E72 success first={first['success']:.4%}，{'last' if complete else 'latest_partial'}={last['success']:.4%}，max={best['success']:.4%}，首次best_step={best['step']}。",
        "", "训练时只在阶段起点成功率至少60%的0/3/5类别约束动作分布。教师仅用于训练损失，运行时仍是单个网络输出完整目标，没有参考动作相加。", "",
        "![六类朝向](stage_retention_modes.png)", "", "![总体与全部失败穿透](stage_retention_overall.png)"]
    if complete:
        by_seed = {e["seed"]:e for e in last["episodes"]}
        lines += ["", "| 末步相对哪一个起点/控制 | 新增成功 | 丢失成功 | 净变化 |",
                  "|---|---:|---:|---:|"]
        for name, episodes, _ in comparisons[:3]:
            gain = sum(by_seed[e["seed"]]["success30"] and not e["success30"] for e in episodes)
            lost = sum(not by_seed[e["seed"]]["success30"] and e["success30"] for e in episodes)
            lines.append(f"| {name} | {gain} | {lost} | {gain-lost:+d}/64 |")
    lines += ["", "穿透图包含全部失败，不只是成功案例；3mm线是主成功门槛，不代表模型所有动作均满足这一门槛。该单次开发结果不能证明换初态、换训练种子后仍提高。", "",
              "原始逐例数据在上一级E46/E72各step目录，曲线数字见stage_retention_curves.csv，复算入口scripts/plot_s09_stage_retention.py。"]
    (output/"stage_retention_results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(report=str(output), points={name:len(rows) for name, _, _, _, rows, _ in series})))


if __name__ == "__main__":
    main()
