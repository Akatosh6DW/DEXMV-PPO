"""Aggregate every frozen cylinder case and draw descriptive, seed-aware figures."""

import csv
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/orca_cylinder_study"
LABELS = {"ppo":"PPO", "guided":"BC + PPO", "residual":"Reference + residual PPO",
          "bc":"BC only", "zero":"Reference only"}
COLORS = {"ppo":"#4676a9", "guided":"#159579", "residual":"#c8653b", "bc":"#898989", "zero":"#9b7cba"}


def write_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    audit = json.loads((OUT / "audit.json").read_text())
    if not audit["complete"] or audit["test_episodes"] != 4224:
        raise ValueError("Complete audit required before final reporting")
    report = OUT / "report"
    report.mkdir(exist_ok=True)
    episodes, runs, curves = [], [], []
    for path in protocol["train_jobs"]:
        job = json.loads((ROOT / path).read_text())
        run = ROOT / job["output"]
        with (run / "metrics.csv").open() as stream:
            metrics = list(csv.DictReader(stream))
        for row in metrics:
            curves.append(dict(method=job["method"], seed=job["seed"], step=int(row["step"]),
                               clean_success30=float(row["clean_success30"]), success30=float(row["success30"]),
                               mean_hold=float(row["mean_hold"]), elapsed_sec=float(row["elapsed_sec"])))
    for path in protocol["eval_jobs"]:
        job = json.loads((ROOT / path).read_text())
        run = ROOT / job["output"]
        data = json.loads((run / "step_000000000/episodes.json").read_text())
        common = dict(method=job["method"], train_seed=job["seed"], budget=job["budget"], condition=job["benchmark"])
        episodes.extend(dict(common, **e) for e in data)
        runs.append(dict(common, successes=sum(e["clean_success30"] for e in data), episodes=len(data),
                         clean_success30=float(np.mean([e["clean_success30"] for e in data])),
                         success30=float(np.mean([e["success30"] for e in data])),
                         success15=float(np.mean([e["success15"] for e in data])),
                         whole_episode_clean=float(np.mean([e["whole_episode_clean_success30"] for e in data])),
                         mean_hold=float(np.mean([e["longest_hold_sec"] for e in data])),
                         drop_rate=float(np.mean([e["drop"] for e in data])),
                         forbidden_rate=float(np.mean([e["substep_forbidden"] for e in data])),
                         penetration_failure_rate=float(np.mean([e["post_release_penetration_m"] > .003 for e in data])),
                         max_depth_mm=1000*max(e["post_release_penetration_m"] for e in data),
                         mean_peak_force_n=float(np.mean([e["substep_force_peak_n"] for e in data])),
                         mean_residual_rms_rad=float(np.mean([e["residual_rms_rad"] for e in data]))))
    assert len(episodes) == 4224 and len(runs) == 66 and len(curves) == 45
    write_csv(report / "episodes.csv", episodes)
    write_csv(report / "runs.csv", runs)
    write_csv(report / "learning_curves.csv", curves)
    groups = []
    for condition, method, budget in sorted({(r["condition"],r["method"],r["budget"]) for r in runs}):
        group = [r for r in runs if (r["condition"],r["method"],r["budget"]) == (condition,method,budget)]
        rates = [r["clean_success30"] for r in group]
        groups.append(dict(condition=condition,method=method,budget=budget,
                           training_seeds=0 if method == "zero" else len(group),
                           successes=sum(r["successes"] for r in group), episodes=sum(r["episodes"] for r in group),
                           mean_clean_success=float(np.mean(rates)), seed_std=float(np.std(rates, ddof=1)) if len(rates)>1 else None,
                           min_seed_success=min(rates), max_seed_success=max(rates),
                           mean_raw_success=float(np.mean([r["success30"] for r in group])),
                           drop_rate=float(np.mean([r["drop_rate"] for r in group])),
                           forbidden_rate=float(np.mean([r["forbidden_rate"] for r in group])),
                           penetration_failure_rate=float(np.mean([r["penetration_failure_rate"] for r in group])),
                           mean_hold=float(np.mean([r["mean_hold"] for r in group])),
                           mean_peak_force_n=float(np.mean([r["mean_peak_force_n"] for r in group]))))
    write_csv(report / "groups.csv", groups)
    pairs = []
    for condition in protocol["conditions"]:
        for budget in protocol["budgets"]:
            for method in ("guided", "residual"):
                differences = []
                for seed in protocol["train_seeds"]:
                    lookup = lambda m: next(r["clean_success30"] for r in runs if
                                            (r["condition"],r["budget"],r["train_seed"],r["method"]) == (condition,budget,seed,m))
                    differences.append(lookup(method)-lookup("ppo"))
                d = np.array(differences)
                permutations = [abs(np.mean(d*np.array(signs))) for signs in itertools.product((-1,1),repeat=3)]
                pairs.append(dict(condition=condition,budget=budget,method=method,
                                  mean_difference=float(d.mean()), seed_differences=differences,
                                  paired_signflip_p=float(np.mean(np.array(permutations)>=abs(d.mean())-1e-12))))
    (report / "paired_descriptive.json").write_text(json.dumps(pairs,indent=2)+"\n")
    # Compare each learned route to its own non-RL control, not just to PPO.
    contributions = []
    for condition in protocol["conditions"]:
        for budget in protocol["budgets"]:
            for method, control in (("guided", "bc"), ("residual", "zero")):
                for seed in protocol["train_seeds"]:
                    learned = next(r for r in runs if (r["condition"], r["budget"], r["train_seed"], r["method"])
                                   == (condition, budget, seed, method))
                    baseline = next(r for r in runs if r["condition"] == condition and r["method"] == control
                                    and (control == "zero" or r["train_seed"] == seed))
                    contributions.append(dict(condition=condition, budget=budget, method=method,
                                              control=control, train_seed=seed,
                                              learned_success=learned["clean_success30"],
                                              control_success=baseline["clean_success30"],
                                              difference=learned["clean_success30"]-baseline["clean_success30"]))
    write_csv(report / "rl_contribution.csv", contributions)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig, ax = plt.subplots(figsize=(8,4.5))
    for method in ("ppo","guided","residual"):
        steps=sorted({r["step"] for r in curves})
        array=np.array([[r["clean_success30"] for r in curves if r["method"]==method and r["step"]==s] for s in steps])
        mean,std=array.mean(1)*100,array.std(1,ddof=1)*100
        ax.plot(steps,mean,label=LABELS[method],color=COLORS[method],marker="o")
        ax.fill_between(steps,np.maximum(0,mean-std),np.minimum(100,mean+std),color=COLORS[method],alpha=.12)
    ax.set(xlabel="Online environment steps",ylabel="Development clean success (%)",ylim=(-2,102),title="Cylinder learning curves: mean +/- seed SD (3 seeds)")
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(report/"learning_curves.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,budget in zip(axes,protocol["budgets"]):
        for i,method in enumerate(("ppo","guided","residual","bc","zero")):
            b=budget if method in ("ppo","guided","residual") else 0
            group=[r for r in runs if r["condition"]=="B0_nominal" and r["method"]==method and r["budget"]==b]
            values=np.array([r["clean_success30"] for r in group])*100
            ax.bar(i,values.mean(),color=COLORS[method],alpha=.65)
            ax.scatter(i+np.linspace(-.1,.1,len(values)),values,color="black",s=20)
        ax.set_xticks(range(5),["PPO","BC+PPO","Residual","BC only","Ref only"],rotation=20)
        ax.set(title=f"{budget:,} online steps",ylim=(-3,105),ylabel="Test clean success (%)")
    fig.suptitle("Nominal task; dots: training seeds, reference: one fixed controller")
    fig.tight_layout();fig.savefig(report/"nominal_budgets.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    methods=("ppo","guided","residual","bc","zero")
    matrix=np.array([[next(g["mean_clean_success"] for g in groups if
                           (g["condition"],g["method"],g["budget"])==(c,m,655360 if m in methods[:3] else 0))
                      for c in protocol["conditions"]] for m in methods])*100
    im=ax.imshow(matrix,cmap="YlGnBu",vmin=0,vmax=100,aspect="auto")
    for (i,j),value in np.ndenumerate(matrix):ax.text(j,i,f"{value:.1f}%",ha="center",va="center",color="white" if value>65 else "black")
    ax.set_xticks(range(3),["Nominal +/-2mm","Position +/-6mm","Push 0.15N"])
    ax.set_yticks(range(5),[LABELS[m] for m in methods]);ax.set_title("Long-budget clean success; unchanged policies")
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(report/"robustness.png",dpi=160);plt.close(fig)
    summary=dict(complete=True,training_runs=9,test_jobs=66,test_episodes=4224,groups=groups,
                 comparison_unit="3 independent training seeds, not 192 independent trainings",
                 significance_note="With 3 paired seeds, two-sided sign-flip p cannot establish p<0.05; descriptive only")
    (report/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["# 圆柱完整实验数据汇总","","正式矩阵已完成：9 次训练，66 项测试，4224 回合。", "",
           "[通俗总结、方法解释、结论与局限](../../../docs/experiments/cylinder_study_20261009.md)", "",
           "以下百分比是三个训练种子的均值；仅参考没有训练种子重复。成功标准未降低。", "",
           "| 条件 | 方法 | PPO步数 | 清洁成功率 | 种子最低/最高 | 原始成功率 | 最长保持均值秒 |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for g in groups:
        lines.append(f"| {g['condition']} | {LABELS[g['method']]} | {g['budget']} | {100*g['mean_clean_success']:.2f}% | "
                     f"{100*g['min_seed_success']:.2f}% / {100*g['max_seed_success']:.2f}% | {100*g['mean_raw_success']:.2f}% | {g['mean_hold']:.2f} |")
    lines += ["", "## 怎样读这些数据", "",
              "清洁成功不仅要求抓住 30 秒、末尾仍有效，还排除撤托后禁用接触和超过 3 mm 的穿透。",
              "原始成功只统计原任务的有效抓持条件，因此可能高于清洁成功。两者都保留。",
              "最长保持均值包括最后掉落的回合，不能把它当成功率。接触力来自仿真，不是硬件载荷认证。",
              "三个种子的离散程度很重要；只有一个种子成功不代表可靠。当前规模不支持强统计优越性结论。",
              "不同初始位置、外力的结果属于鲁棒性扩展；不根据哪一项胜出重新挑主任务。", "",
              "![学习曲线](learning_curves.png)", "![名义条件两预算](nominal_budgets.png)", "![鲁棒性](robustness.png)", "",
              "逐回合见 episodes.csv，各种子见 runs.csv，汇总见 groups.csv；配对差异见 paired_descriptive.json。",
              "rl_contribution.csv 单独记录 BC后PPO减仅BC、残差减仅参考；差值为负表示这一轮强化学习造成退化。",
              "groups.csv 另列掉落、禁用接触和穿透超限比例；这些失败可同时发生，不能直接相加当作总失败率。",
              "视频参考来自方块视频并经过圆柱仿真校正，不是匹配圆柱视频提取。BC 数据是 29 条成功仿真回放，不是 29 个独立视频。",
              "训练步数相等不等于总成本相等：参考搜索、示范采集和 BC 都有额外成本。"]
    (report/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(complete=True,groups=len(groups),episodes=len(episodes),report=str(report))))


if __name__ == "__main__":
    main()
