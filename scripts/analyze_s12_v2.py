"""Summarize completed and partial V2 development runs without using V1 data."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/orca_benchmark_suite/S12"


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    output = BASE/"v2_development_report"
    output.mkdir(exist_ok=True)
    probes, curves, summaries, episodes = [], [], [], []
    for path in sorted(BASE.glob("V2_*")):
        report = path/"report.json"
        if report.exists() and path.name.startswith("V2_P"):
            r = json.loads(report.read_text())
            probes.append(dict(run=path.name, offset_z_mm=1000*r["offset"][2],
                sectors=r["config"]["sectors"], timestep_ms=1000*r["config"]["timestep"],
                grip_scale=r["grip_scale"], success15=r["success15"], hold_sec=r["best_hold_sec"],
                strain_pct=100*r["max_principal_strain"], penetration_mm=1000*r["max_penetration_m"],
                force_n=r["peak_force_n"], volume_min=r["min_volume_ratio"],
                final_z=r["final_center"][2]))
        if (path/"metrics.csv").exists():
            with (path/"metrics.csv").open() as f:
                rows = [{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
            completed = (path/"summary.json").exists()
            config = json.loads((path/"run_config.json").read_text())
            records = path/"step_000000000/episodes.json"
            insufficient = (config["method"]=="collect" and records.exists()
                            and sum(e["success15"] for e in json.loads(records.read_text()))<4)
            status = "completed" if completed else "dataset_failed" if insufficient else "partial"
            for row in rows:
                curves.append(dict(run=path.name, completed=completed, status=status, **row))
            for metric in ("success15", "hold_sec"):
                best = max(rows, key=lambda r:r[metric])
                summaries.append(dict(run=path.name, completed=completed, status=status, metric=metric,
                    first=rows[0][metric], last=rows[-1][metric], max=best[metric], best_step=best["step"]))
        for file in sorted(path.glob("step_*/episodes.json")):
            for episode in json.loads(file.read_text()):
                episodes.append(dict(run=path.name, step=int(file.parent.name[5:]), **episode))
    write_csv(output/"physical_probes.csv", probes)
    write_csv(output/"learning_curves.csv", curves)
    write_csv(output/"first_last_max.csv", summaries)
    write_csv(output/"episodes.csv", episodes)
    (output/"data.json").write_text(json.dumps(dict(probes=probes, curves=curves,
        summaries=summaries, episodes=episodes), indent=2)+"\n")

    material = BASE/"V2_M01_compression_dt02ms"
    if (material/"report.json").exists():
        material_rows = json.loads((material/"report.json").read_text())
        fig,axes = plt.subplots(1,2,figsize=(10,3.8))
        tab = []
        for row in material_rows:
            young = row["config"]["young"]
            data = np.genfromtxt(material/f"E{int(young)}_compression.csv",delimiter=",",names=True)
            axes[0].plot(data["time"],data["normal_force_n"],label=f"{young/1000:g} kPa")
            axes[1].plot(data["time"],100*data["max_principal_strain"],label=f"{young/1000:g} kPa")
            tab.append(dict(young_kpa=young/1000,steady_force_n=row["compression_force_n"],
                loaded_strain=row["loaded_strain"],recovery_strain=row["recovery_strain"],
                min_volume_ratio=row["min_volume_ratio"]))
        for ax,label in zip(axes,("Plate normal force (N)","Maximum principal strain (%)")):
            ax.set(xlabel="Time (s)",ylabel=label)
            ax.axvspan(.4,.7,color="gray",alpha=.1)
            ax.legend();ax.grid(alpha=.2)
        fig.suptitle("Independent material check: 5% compression, hold, unload")
        fig.tight_layout();fig.savefig(output/"material_check.png",dpi=160);plt.close(fig)
        write_csv(output/"material_check.csv",tab)

    fig, axes = plt.subplots(1,3, figsize=(12,3.8))
    selected = [r for r in probes if r["grip_scale"]==1 and r["sectors"]==12 and r["timestep_ms"]==.2]
    selected.sort(key=lambda r:r["offset_z_mm"])
    x = [r["offset_z_mm"] for r in selected]
    for ax,field,label in zip(axes,("hold_sec", "strain_pct", "penetration_mm"),
                              ("Strict continuous hold (s)", "Peak strain (%)", "Peak penetration (mm)")):
        ax.plot(x,[r[field] for r in selected], "o-")
        ax.set(xlabel="Hand height offset (mm)", ylabel=label)
        ax.grid(alpha=.2)
    axes[0].axhline(15, color="gray", linestyle="--")
    fig.suptitle("S12 V2 development: one-factor placement probes, not test-set accuracy")
    fig.tight_layout()
    fig.savefig(output/"placement_probes.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1,2, figsize=(11,4))
    for run in sorted({r["run"] for r in curves if r["run"].startswith("V2_T")}):
        rows = [r for r in curves if r["run"]==run]
        label = run+(" [partial]" if not rows[-1]["completed"] else "")
        for ax, metric in zip(axes,("success15", "hold_sec")):
            ax.plot([r["step"] for r in rows], [r[metric] for r in rows], "o-", label=label)
            ax.set(xlabel="Online environment steps", ylabel=metric)
            ax.grid(alpha=.2)
    if any(r["run"].startswith("V2_T") for r in curves):
        axes[0].legend(fontsize=7)
    axes[0].set_ylim(-.05,1.05)
    fig.suptitle("Development states only; no claim about held-out performance")
    fig.tight_layout()
    fig.savefig(output/"development_learning.png", dpi=160)
    plt.close(fig)
    lines = ["# S12 V2 开发数据", "", "这些是开发探测，不是最终测试集结果。V1 网格数据已排除。",
        "进行中的训练明确标为 partial；表格不是只筛选成功案例。", "",
        "## 材料独立检查", "", "![压缩和回弹](material_check.png)", "",
        "压缩量固定为初始高度的 5%，0.4--0.7 秒保持，然后卸载。不同刚度的稳态反力分别约 0.513/0.979/1.872 N，卸载后恢复。",
        "这支持弹性形变与刚度趋势的合理性，不是实物验证，也不证明复杂抓取中的网格已收敛。初始接触力瞬态不作为稳态反力。", "",
        "## 摆位和物理检查", "", "![摆位扫描](placement_probes.png)", "",
        "| 探测 | 手位 z/mm | 圆周分段 | 步长/ms | 严格成功 | 保持/s | 最大应变/% |",
        "|---|---:|---:|---:|---|---:|---:|"]
    for r in probes:
        lines.append(f"| {r['run']} | {r['offset_z_mm']:.1f} | {r['sectors']} | {r['timestep_ms']:.2f} | {r['success15']} | {r['hold_sec']:.2f} | {r['strain_pct']:.2f} |")
    lines += ["", "严格成功还要求撤支撑后的接触始终合格，因此最长保持超过 15 秒也可能失败；不是计算矛盾。",
        "细网格下局部接触可能改变，不能只看到粗网格成功就写已经数值收敛。", "",
        "## 训练开发曲线", "", "![开发曲线](development_learning.png)", "",
        "| 运行 | 状态 | 指标 | first | last | max | best_step |", "|---|---|---|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['run']} | {r['status']} | {r['metric']} | {r['first']:.4f} | {r['last']:.4f} | {r['max']:.4f} | {r['best_step']:.0f} |")
    lines += ["", "## 原始记录", "", "- `episodes.csv`：全部已保存的逐回合评估，不把重复初态当独立新样本。",
        "- `learning_curves.csv`、`first_last_max.csv`：训练各阶段和首末/峰值统计。",
        "- `physical_probes.csv`：完成的原生物理探测；数值崩溃、未完成探测没有 report，详见开发日志。",
        "- 每个 V2 运行目录保留配置、模型、轨迹、已完成训练的权重与源码快照。",
        "- 文献、判据及限制见 `docs/experiments/S12/README.md`；本页由 `scripts/analyze_s12_v2.py` 生成。"]
    (output/"results.md").write_text("\n".join(lines)+"\n")
    print(output)


if __name__ == "__main__":
    main()
