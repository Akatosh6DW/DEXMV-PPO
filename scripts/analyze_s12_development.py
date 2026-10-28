"""Generate evidence-linked S12 development tables and diagnostic figures."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from s12_soft import OUT


def longest(mask):
    current = best = 0
    for flag in mask:
        current = current+1 if flag else 0
        best = max(best, current)
    return best*.05


def main():
    report = OUT/"development_report"
    report.mkdir(exist_ok=True)
    material = OUT/"M02_compression_dt01ms"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    results = json.loads((material/"report.json").read_text())
    axes[0].plot([r["config"]["young"]/1000 for r in results],
                 [r["compression_force_n"] for r in results], "o-")
    axes[0].set(xlabel="Young's modulus (kPa)", ylabel="Normal force (N)", title="5% compression")
    for r in results:
        y = int(r["config"]["young"])
        data = np.genfromtxt(material/f"E{y}_compression.csv", delimiter=",", names=True)
        axes[1].plot(data["time"], data["max_principal_strain"]*100, label=f"{y//1000} kPa")
    axes[1].set(xlabel="Time (s)", ylabel="Max local strain (%)", title="Compression and recovery")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(report/"material_validation.png", dpi=160); plt.close(fig)

    demo = OUT/"D02_soft_demonstrations/step_000000000"
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for path in sorted(demo.glob("trajectory_*.npz")):
        with np.load(path) as z:
            center = z["vertices"].mean(1)
            paired = (z["loads"][:, 0]>.02)&(z["loads"][:, 1:].max(1)>.02)
            supported = z["support_force_n"]>1e-5
            physical = (center[:, 2]>=.535)&paired&~supported&(z["time"]>=5.5)
            rows.append(dict(seed=int(path.stem.split("_")[-1]),
                physical_hold_sec=longest(physical), physical_success15=longest(physical)>=15,
                strict_success15=bool(z["success15"][-1]), strict_best_sec=float(z["hold_sec"].max()),
                strict_final_sec=float(z["hold_sec"][-1]), strain_max=float(z["max_principal_strain"].max()),
                depth_max_m=float(z["penetration_m"].max()), undesired_contact_max_n=float(z["forbidden_force_n"].max()),
                peak_single_contact_n=float(z["peak_force_n"].max()), source=str(path.relative_to(OUT))))
            if rows[-1]["seed"] in (250100, 250103):
                label=str(rows[-1]["seed"])
                axes[0].plot(z["time"], (center[:, 2]-.5)*1000, label=label)
                axes[1].plot(z["time"], z["forbidden_force_n"], label=label)
    axes[0].axhline(35, color="gray", linestyle=":")
    axes[0].set(ylabel="Center lift (mm)", title="Both objects lift; contact quality separates outcomes")
    axes[0].legend()
    axes[1].axhline(.02, color="gray", linestyle=":")
    axes[1].set(xlabel="Time (s)", ylabel="Undesired contact load (N)")
    fig.tight_layout(); fig.savefig(report/"hold_vs_contact_quality.png", dpi=160); plt.close(fig)
    if rows:
        with (report/"demonstration_episodes.csv").open("w") as f:
            writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    training = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for path in sorted(OUT.glob("T*/metrics.csv")):
        with path.open() as f:
            curve = [{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
        if not curve:
            continue
        completed = (path.parent/"summary.json").exists()
        training.append(dict(run=path.parent.name, completed=completed, rows=curve))
        label = path.parent.name+("" if completed else " (partial)")
        axes[0].plot([r["step"] for r in curve], [100*r["success15"] for r in curve], "o-", label=label)
        axes[1].plot([r["step"] for r in curve], [r["hold_sec"] for r in curve], "o-", label=label)
    axes[0].set(xlabel="Online interaction steps", ylabel="Development success (%)", ylim=(-5, 105))
    axes[1].set(xlabel="Online interaction steps", ylabel="Mean best strict hold (s)")
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(report/"development_learning.png", dpi=160); plt.close(fig)
    payload = dict(material=results, demonstration_episodes=rows, training=training)
    (report/"data.json").write_text(json.dumps(payload, indent=2)+"\n")
    text = ["# S12 V1 开发数据报告（网格缺陷，不能作为最终结果）", "",
        "V1 Delaunay 网格有额外内部边界面。本页仅追踪缺陷发现之前的开发过程，全部物理结论需 V2 一致网格重新验证。",
        "", "## 材料验证", "", "![压缩与回弹](material_validation.png)",
        "", "同样压缩量，材料越硬所需反力越大；卸载后应变回到接近零。详细值来自 M02 的原始 CSV。",
        "", "## 示范采集", "", "![保持与接触](hold_vs_contact_quality.png)", "",
        "物理保持：已抬升至少 35 mm，拇指与其他手指指腹承载、无托台支撑，连续 15 秒。",
        "严格成功还要求形变、穿透、载荷及指定接触面全满足条件，并且撤支撑后没有出现违规接触。",
        "", "| 初态 | 物理保持秒数 | 严格最佳秒数 | 严格成功 | 最大形变 | 非指定接触 N |",
        "|---|---:|---:|---|---:|---:|"]
    for r in rows:
        text.append(f"| {r['seed']} | {r['physical_hold_sec']:.2f} | {r['strict_best_sec']:.2f} | "
                    f"{r['strict_success15']} | {100*r['strain_max']:.2f}% | {r['undesired_contact_max_n']:.4f} |")
    if rows:
        text += ["", f"全部 {len(rows)} 次尝试中，物理保持通过 {sum(r['physical_success15'] for r in rows)} 次，"
                 f"严格成功 {sum(r['strict_success15'] for r in rows)} 次。只用严格成功的样本拟合 BC，未删除失败记录。"]
    text += ["", "## 学习曲线", "", "![开发曲线](development_learning.png)", "",
             "图中 partial 是尚未结束的训练，只能读已完成的评估点。不同初始化的 0 步结果不能归功于 PPO。",
             "", "原始汇总：`data.json`；逐初态数据：`demonstration_episodes.csv`。"]
    (report/"results.md").write_text("\n".join(text)+"\n")
    print(json.dumps(dict(episodes=len(rows), training_runs=len(training), report=str(report)), ensure_ascii=False))


if __name__ == "__main__":
    main()
