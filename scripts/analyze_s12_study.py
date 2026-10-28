"""Paired multi-seed S12 statistics, plots and plain-language report from frozen outputs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

LABELS = {"ppo":"Pure PPO", "guided":"BC + PPO", "residual":"Reference + residual PPO",
          "reference":"Reference replay", "bc_initial":"BC initial"}
ORDER = ("ppo", "bc_initial", "guided", "reference", "residual")
COLORS = dict(zip(ORDER, ("#777777", "#3b82b6", "#16816a", "#b57d22", "#b64b5d")))


def csv_write(path, rows):
    if rows:
        with path.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader(); writer.writerows(rows)


def hierarchical_interval(values, seed=120, repeats=10000):
    """Resample training policies and paired initial states, not flattened episodes."""
    a = np.asarray(values, float)
    if a.ndim!=2 or not a.size:
        raise ValueError("Expected [policy_seed, initial_state]")
    rng = np.random.default_rng(seed)
    rows = rng.integers(a.shape[0], size=(repeats,a.shape[0]))
    columns = rng.integers(a.shape[1], size=(repeats,a.shape[1]))
    estimates = a[rows[:,:,None], columns[:,None,:]].mean((1,2))
    return np.quantile(estimates,[.025,.975]).tolist()


def wilson(successes, count):
    z = 1.959963984540054
    p = successes/count
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    half = z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [max(0.,center-half),min(1.,center+half)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    a = parser.parse_args()
    plan = json.loads((a.study/"protocol.json").read_text())
    output = a.study/"report"
    output.mkdir(exist_ok=True)
    curves, training_summary, rows = [], [], []
    missing = []
    for job in plan["jobs"]:
        run = Path(job["output"])
        metrics = run/"metrics.csv"
        if not metrics.exists():
            missing.append(job["name"]+": no training metrics")
            continue
        with metrics.open() as f:
            data = [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
        summary = json.loads((run/"summary.json").read_text()) if (run/"summary.json").exists() else {}
        complete = (summary.get("status")=="completed" and data[-1]["step"]==plan["online_steps"]
                    and summary["rows"][-1]["step"]==plan["online_steps"])
        if not complete:
            missing.append(job["name"]+": training incomplete")
        curves += [dict(method=job["method"], train_seed=job["seed"], completed=complete, **r) for r in data]
        for metric in ("success15", "hold_sec", "strain_max"):
            best = (min if metric=="strain_max" else max)(data,key=lambda r:r[metric])
            training_summary.append(dict(method=job["method"],train_seed=job["seed"],metric=metric,
                first=data[0][metric],last=data[-1][metric],best=best[metric],best_step=best["step"],completed=complete))
    evaluation_file = a.study/"evaluation_plan.json"
    if evaluation_file.exists():
        ep = json.loads(evaluation_file.read_text())
        if ep["protocol_sha256"]!=hashlib.sha256((a.study/"protocol.json").read_bytes()).hexdigest():
            raise ValueError("Protocol hash mismatch")
        for job in ep["jobs"]:
            run = Path(job["output"])
            if not (run/"summary.json").exists():
                missing.append(job["name"]+": evaluation incomplete")
                continue
            config = json.loads((run/"evaluation_config.json").read_text())
            data = json.loads((run/"episodes.json").read_text())
            summary = json.loads((run/"summary.json").read_text())
            if (summary.get("status")!="completed" or summary["count"]!=job["eval_count"]
                    or [r["seed"] for r in data]!=plan["conditions"][job["condition"]]["seeds"]
                    or config["checkpoint_sha256"]!=job["checkpoint_sha256"]
                    or config["condition"]!=job["condition"]):
                raise ValueError("Evaluation binding mismatch: "+job["name"])
            rows += [dict(method=job["label"], train_seed=job["train_seed"], condition=job["condition"],
                          trace=str(run/f"trajectory_{r['seed']}.npz"), **r) for r in data]
    else:
        missing.append("Evaluation not yet bound or run")
    if missing and not a.allow_partial:
        raise ValueError("Not a completed study: "+"; ".join(missing))
    csv_write(output/"learning_curves.csv",curves)
    csv_write(output/"training_first_last.csv",training_summary)
    csv_write(output/"episodes.csv",rows)
    aggregate, individual, differences = [], [], []
    matrices = {}
    for condition in plan["conditions"]:
        for method in ORDER:
            expected_sources = [None] if method=="reference" else plan["train_seeds"]
            group = [r for r in rows if r["condition"]==condition and r["method"]==method]
            if not group:
                continue
            records = [[r for r in group if r["train_seed"]==s] for s in expected_sources]
            if any(len(r)!=len(plan["conditions"][condition]["seeds"]) for r in records):
                continue
            values = np.array([[r["success15"] for r in record] for record in records],float)
            matrices[condition,method] = values
            low, high = hierarchical_interval(values)
            aggregate.append(dict(condition=condition,method=method,policies=len(expected_sources),
                states_per_policy=values.shape[1],success_mean=float(values.mean()),ci_low=low,ci_high=high,
                physical_success_mean=float(np.mean([r["physical_success15"] for r in group])),
                hold_mean=float(np.mean([r["best_hold_sec"] for r in group])),
                strain_mean=float(np.mean([r["max_principal_strain"] for r in group])),
                strain_max=max(r["max_principal_strain"] for r in group),
                penetration_max=max(r["max_penetration_m"] for r in group),
                force_max=max(r["peak_force_n"] for r in group),
                element_collapse_rate=float(np.mean([r["min_volume_ratio"]<=.05 for r in group])),
                element_inversion_rate=float(np.mean([r["min_volume_ratio"]<=0 for r in group])),
                requested_residual_rms_mean=float(np.mean([r["requested_residual_rms"] for r in group]))))
            for source, record in zip(expected_sources, records):
                successes = sum(r["success15"] for r in record)
                low,high = wilson(successes,len(record))
                individual.append(dict(condition=condition,method=method,train_seed=source,
                    successes=successes,count=len(record),success_rate=successes/len(record),
                    wilson_low=low,wilson_high=high))
        for left,right in (("guided","bc_initial"),("residual","reference"),
                           ("guided","ppo"),("residual","ppo")):
            if (condition,left) in matrices and (condition,right) in matrices:
                difference = matrices[condition,left]-matrices[condition,right]
                low,high = hierarchical_interval(difference)
                differences.append(dict(condition=condition,left=left,right=right,
                    difference=float(difference.mean()),ci_low=low,ci_high=high))
    csv_write(output/"aggregate.csv",aggregate)
    csv_write(output/"per_training_seed.csv",individual)
    csv_write(output/"paired_differences.csv",differences)
    payload = dict(status="partial" if missing else "completed", missing=missing,
        aggregate=aggregate, individual=individual, differences=differences)
    (output/"statistics.json").write_text(json.dumps(payload,indent=2)+"\n")

    fig,axes = plt.subplots(1,2,figsize=(11,4))
    styles = dict(zip(plan["train_seeds"], ("-", "--", ":")))
    for method in ("ppo","guided","residual"):
        for source in plan["train_seeds"]:
            data = [r for r in curves if r["method"]==method and r["train_seed"]==source]
            for ax,metric in zip(axes,("success15","hold_sec")):
                ax.plot([r["step"] for r in data],[r[metric] for r in data],color=COLORS[method],alpha=.8,
                    linestyle=styles[source],
                    label=LABELS[method] if source==plan["train_seeds"][0] else None)
                ax.set(xlabel="Online training steps",ylabel=metric)
                ax.grid(alpha=.2)
    axes[0].set_ylim(-.05,1.05)
    axes[0].set_ylabel("Strict success (fraction)")
    axes[1].set_ylabel("Mean best strict hold (s)")
    axes[0].legend(fontsize=8,loc="lower left",bbox_to_anchor=(.02,.05))
    axes[1].legend(handles=[Line2D([0],[0],color="black",linestyle=styles[s],label=f"Seed {s}")
                           for s in plan["train_seeds"]],fontsize=8)
    fig.suptitle("Development learning curves: each line is one training seed")
    fig.tight_layout(); fig.savefig(output/"learning_curves.png",dpi=160); plt.close(fig)
    if aggregate:
        fig,axes = plt.subplots(1,2,figsize=(11,4))
        nominal = [r for r in aggregate if r["condition"]=="nominal"]
        for i,r in enumerate(nominal):
            axes[0].bar(i,100*r["success_mean"],color=COLORS[r["method"]])
            axes[0].errorbar(i,100*r["success_mean"],yerr=[[100*(r["success_mean"]-r["ci_low"])],
                [100*(r["ci_high"]-r["success_mean"])]],fmt="none",ecolor="black",capsize=3)
            for item in individual:
                if item["condition"]=="nominal" and item["method"]==r["method"]:
                    axes[0].plot(i,100*item["success_rate"],"k.")
            axes[1].bar(i,100*r["physical_success_mean"],color=COLORS[r["method"]])
        for ax in axes:
            ax.set_xticks(range(len(nominal)),[LABELS[r["method"]] for r in nominal],rotation=20,ha="right",fontsize=8)
            ax.set_ylim(0,105); ax.set_ylabel("Success (%)")
        axes[0].set_title("Strict contact/material gate")
        axes[1].set_title("Retention with paired pad loads")
        fig.tight_layout();fig.savefig(output/"nominal_success.png",dpi=160);plt.close(fig)
        fig,ax = plt.subplots(figsize=(9,4))
        for method in ORDER:
            data = [next((r for r in aggregate if r["condition"]==c and r["method"]==method),None) for c in plan["conditions"]]
            ax.plot(range(len(data)),[100*r["success_mean"] if r else np.nan for r in data],"o-",label=LABELS[method],color=COLORS[method])
        ax.set_xticks(range(len(plan["conditions"])),list(plan["conditions"]))
        ax.set(ylabel="Strict success (%)",ylim=(-5,105))
        ax.legend(fontsize=8,loc="upper center",bbox_to_anchor=(.5,1.22),ncol=3);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(output/"transfer.png",dpi=160);plt.close(fig)

    lines = ["# S12 软体抓取实验统计", "", "**状态："+("尚未完成，不得作为最终结论。" if missing else "预定训练和评估数据已齐全。")+"**",
        "", "## 这次到底比较什么", "", "同一个可变形圆柱、同一手部接近/抬升路径下，比较纯 PPO、BC 后 PPO、固定参考、参考加残差 PPO。BC 起点单列，避免把模仿学到的能力算成 PPO 改进。",
        "每个学习方法三个训练种子，各 51200 在线步；统一评估末步。原有视频几何参考复用，但在软体环境重新采集示范，不是新 RGB 提取，也不是现实软体材料标定。",
        "", "## 学习过程", "", "![开发曲线](learning_curves.png)", "", "开发集曲线可用于理解过程，最终表用独立初态，不能把两者混起来。完整首末、峰值和逐训练种子记录在 CSV。", "",
        "## 独立测试", "", "| 条件 | 方法 | 严格成功率 | 物理保持成功率 | 平均最佳保持/s | 最大应变/% | 最大穿透/mm | 单元塌陷率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in aggregate:
        lines.append(f"| {r['condition']} | {LABELS[r['method']]} | {100*r['success_mean']:.2f}% | {100*r['physical_success_mean']:.2f}% | {r['hold_mean']:.2f} | {100*r['strain_max']:.2f} | {1000*r['penetration_max']:.3f} | {100*r['element_collapse_rate']:.2f}% |")
    lines += ["", "物理保持仍要求撤支撑、抬升、拇指与另一手指指定正面持续承载十五秒，只不施加严格质量门槛；它不是单纯未掉落。",
        "单元塌陷指局部体积比 <=0.05；其中非正值表示单元翻转，已超出有效材料模拟范围。这些回合仍计失败、保留在分母内，不当成正常掉落或真实材料损坏。出现塌陷时，本表描述的是策略与这套模拟器共同组成的系统表现，不能全部归因为真实抓取能力。具体失败原因另见 `failure_analysis.md`，物理有效性审计见 `verification.json`。"]
    if aggregate:
        lines += ["", "![成功与物理保持](nominal_success.png)", "", "![迁移检查](transfer.png)", "",
            "不同条件使用不同初态种子；曲线描述各条件的保留测试表现，不是同一个初态只改材料/网格的因果试验。方法之间的成对差值仅在同一条件内计算。"]
    lines += ["", "## PPO 真正增加了多少", "", "| 条件 | 对比 | 成功率差/百分点 | 配对分层 bootstrap 95% 区间 |",
        "|---|---|---:|---|"]
    for r in differences:
        lines.append(f"| {r['condition']} | {LABELS[r['left']]} - {LABELS[r['right']]} | {100*r['difference']:+.2f} | [{100*r['ci_low']:+.2f}, {100*r['ci_high']:+.2f}] |")
    lines += ["", "区间跨 0 时，不能说本轮证明了稳定增益。只有三个训练种子，区间属于描述性证据；不能把相同网络在 64 个初态上的结果视为 64 个独立训练实验。参考回放只算一个控制器，不复制成三个独立来源。",
        "全成功/全失败时 bootstrap 区间可能退化，不能解释为没有不确定性；`per_training_seed.csv` 另给每个固定策略的 Wilson 区间。",
        "", "## 不能越过的结论边界", "", "有限预算、单个水平圆柱、仿真状态、理想手部位置控制、未标定弹性材料；不能推广为实机或所有软物体成功率。粗细网格、严格接触与物理保持的差异必须保留。",
        "", "## 数据位置", "", "- `episodes.csv`：逐回合数据和完整轨迹路径。", "- `per_training_seed.csv`：每个独立网络的成功数和分母。",
        "- `paired_differences.csv`：同初态配对差值。", "- `aggregate.csv`：各条件质量指标。",
        "- `learning_curves.csv`、`training_first_last.csv`：训练过程。", "- 上级 `protocol.json` 固定方案，`evaluation_plan.json` 绑定末步权重。",
        "- 原理、术语、文献和开发失败见 `docs/experiments/S12/`。"]
    if missing:
        lines += ["", "## 尚未完成", ""]+["- "+item for item in missing]
    (output/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(status=payload["status"], episodes=len(rows), missing=len(missing), report=str(output)),indent=2))


if __name__ == "__main__":
    main()
