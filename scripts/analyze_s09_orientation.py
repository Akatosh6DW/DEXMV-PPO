"""Descriptive reference-coverage diagnostics, without changing any policy."""

import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/s09-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from analyze_s09_study import LABELS
from run_s09_study import METHODS, OUT, check_protocol


def axis(quaternion):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
    return -matrix.reshape(3, 3)[:, 1]


def main():
    check_protocol()
    report = OUT / "report"
    assert json.loads((report / "matrix_verification.json").read_text())["status"] == "matrix_verified"
    model = mujoco.MjModel.from_xml_path(str(OUT / "inputs/model.xml"))
    oid = int(model.jnt_qposadr[model.joint("cube_freejoint").id])
    with np.load(OUT / "inputs/reference.npz") as bank:
        axes = np.stack([axis(q) for q in bank["object_quaternion"]])
    with np.load(OUT / "eval/uniform/reference/step_000000000/initial_states.npz") as z:
        orientation = np.stack([axis(q[oid+3:oid+7]) for q in z["qpos"]])
        products = np.abs(orientation @ axes.T)
        nearest = products.argmax(1)
        angle = np.rad2deg(np.arccos(np.clip(products.max(1), 0, 1)))
        metadata = {int(seed): dict(template=int(i), nearest_angle_deg=float(a))
                    for seed, i, a in zip(z["seeds"], nearest, angle)}
    with (report / "episodes.csv").open() as handle:
        raw = [dict(r, **metadata[int(r["seed"])]) for r in csv.DictReader(handle) if r["stratum"] == "uniform"]
    for r in raw:
        if r["method"] in ("residual", "reference"):
            assert r["template"] == int(r["selected_reference"])
    groups = []
    for method in (*METHODS, "reference"):
        for kind, bins in (("template", list(range(6))), ("angle_bin", list(range(4)))):
            for index in bins:
                filtered = [r for r in raw if r["method"] == method and
                    (r["template"] == index if kind == "template" else
                     min(int(np.searchsorted([10, 20, 40], r["nearest_angle_deg"], side="right")), 3) == index)]
                per_seed = []
                for seed in sorted({r["training_seed"] for r in filtered}):
                    subset = [r for r in filtered if r["training_seed"] == seed]
                    per_seed.append(float(np.mean([r["success30"] == "True" for r in subset])))
                groups.append(dict(method=method, grouping=kind, index=index, evaluations=len(filtered),
                    unique_initial_states=len({r["seed"] for r in filtered}),
                    success_rate=float(np.mean(per_seed)) if per_seed else None, per_training_seed=per_seed))
    (report / "orientation_coverage.json").write_text(json.dumps(dict(initial_states=metadata, groups=groups), indent=2) + "\n")
    with (report / "orientation_episodes.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    fig, plots = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    colors = ("#777777", "#3c8c70", "#d25c5c", "#4a86b7")
    for j, method in enumerate((*METHODS, "reference")):
        for ax, kind, count in zip(plots, ("template", "angle_bin"), (6, 4)):
            rows = [r for r in groups if r["method"] == method and r["grouping"] == kind]
            values = [np.nan if r["success_rate"] is None else r["success_rate"]*100 for r in rows]
            ax.bar(np.arange(count)+(j-1.5)*.19, values, .19, label=LABELS[method], color=colors[j])
    plots[0].set(xticks=np.arange(6), xticklabels=["T0(X)", "T1(Y)", "T2(Z)", "T3", "T4", "T5"], title="Nearest reference template")
    plots[1].set(xticks=np.arange(4), xticklabels=["0-10", "10-20", "20-40", "40-90"],
                 xlabel="Undirected axis distance (degrees)", title="Distance from nearest reference")
    for ax in plots:
        ax.set(ylim=(0, 100), ylabel="Strict 30s success (%)")
    fig.legend(*plots[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=8)
    fig.savefig(report / "orientation_coverage.png", dpi=160)
    plt.close(fig)
    lines = ["# S09参考朝向覆盖分析", "",
        "仅分析预先冻结的128个连续随机初态，不增加或删除测试。每个初态计算到6个参考模板的最近无向轴夹角；圆柱首尾翻转与绕自身轴转动视为等价。",
        "分箱只用于事后诊断，不是训练课程，也没有给纯PPO/BC后PPO偷偷输入参考。表中学习组按3个训练种子平均，同一初态不会被说成3个不同朝向。", "",
        "![参考覆盖](orientation_coverage.png)", "",
        "| 分组 | 编号 | 方法 | 不同初态数 | 评估次数 | 成功率 |", "|---|---|---|---:|---:|---:|"]
    for r in groups:
        rate = "无样本" if r["success_rate"] is None else f"{r['success_rate']*100:.2f}%"
        lines.append(f"| {r['grouping']} | {r['index']} | {LABELS[r['method']]} | {r['unique_initial_states']} | {r['evaluations']} | {rate} |")
    lines += ["", "template=0/1/2对应X/Y/Z开发模板，3/4/5是三个随机开发模板；angle_bin=0/1/2/3分别对应[0,10)、[10,20)、[20,40)、[40,90]度。",
        "即使角度近也可能因为位置、指腹接触、物体旋转或局部几何失败；夹角与成功率的相关性不是因果证明。某些箱的初态很少，不能强推显著性。",
        "本分析不能证明增加模板一定解决问题，只用于定位应优先检查的方向。所有失败模板仍保留在参考库，未因成绩差而排除。"]
    (report / "orientation_coverage.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(initial_states=len(metadata), groups=len(groups))), flush=True)


if __name__ == "__main__":
    main()
