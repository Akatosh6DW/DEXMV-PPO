"""Aggregate every completed acquisition comparison, clustered by training seed."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np


def write_csv(path, rows):
    if not rows:
        raise ValueError("Empty result table")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def cluster_interval(values):
    values = np.asarray(values, float)
    if len(values) < 2:
        return [None, None]
    rng = np.random.default_rng(44321)
    boot = values[rng.integers(len(values), size=(20000, len(values)))].mean(axis=1)
    return np.quantile(boot, [.025, .975]).tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    expected = {Path(j["name"]).name for j in protocol["jobs"] if j["name"].startswith("eval/")}
    actual = {p.name for p in (root / "eval").iterdir() if p.is_dir()}
    if actual != expected:
        raise ValueError(f"Evaluation matrix differs: missing={expected-actual}, extra={actual-expected}")
    output = root / "report"
    output.mkdir(exist_ok=True)
    episodes, runs, learning = [], [], []
    for directory in sorted((root / "eval").iterdir()):
        if not directory.is_dir():
            continue
        if not (directory / "summary.json").exists():
            raise ValueError("Partial evaluation: " + str(directory))
        meta = json.loads((directory / "run_config.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["status"] == "completed"
        rows = json.loads((directory / "step_000000000/episodes.json").read_text())
        if len(rows) != meta["eval_count"] or [r["seed"] for r in rows] != list(range(meta["eval_start"], meta["eval_start"]+meta["eval_count"])):
            raise ValueError("Missing or reordered evaluation episodes")
        method = "zero_reference" if directory.name.startswith("zero_") else meta["method"]
        scenario = "nominal" if meta["jitter"] == .002 else "offset4mm"
        tag = dict(method=method, scenario=scenario, training_seed=meta["seed"], run=directory.name)
        for row in rows:
            row["clean_success30"] = row["success30"] and not row["forbidden"] and row["max_penetration_m"] <= .003
            row["outcome"] = ("success" if row["success30"] else "never_acquired" if row.get("acquisition_sec") is None
                              else "insufficient_continuous_hold" if row["longest_hold_sec"] < 30
                              else "lost_final_validity")
            episodes.append(dict(tag, **row))
        runs.append(dict(tag, episodes=len(rows), successes=sum(r["success30"] for r in rows),
                         success30=float(np.mean([r["success30"] for r in rows])),
                         success15=float(np.mean([r["success15"] for r in rows])),
                         clean_success30=float(np.mean([r["clean_success30"] for r in rows])),
                         longest_hold_sec=float(np.mean([r["longest_hold_sec"] for r in rows])),
                         finger_peak_n=float(np.mean([r["finger_peak_n"] for r in rows])),
                         forbidden_rate=float(np.mean([r["forbidden"] for r in rows])),
                         max_penetration_mm=1000*max(r["max_penetration_m"] for r in rows)))
    aggregates = []
    for method, scenario in sorted({(r["method"], r["scenario"]) for r in runs}):
        group = [r for r in runs if (r["method"], r["scenario"]) == (method, scenario)]
        counts = [r["episodes"] for r in group]
        assert len(set(counts)) == 1
        values = [r["success30"] for r in group]
        aggregates.append(dict(method=method, scenario=scenario, training_seeds=[r["training_seed"] for r in group],
                               successes=sum(r["successes"] for r in group), episodes=sum(counts),
                               success30=float(np.mean(values)), seed_std=float(np.std(values, ddof=1)) if len(values)>1 else None,
                               seed_bootstrap95=cluster_interval(values), seed_rates=values,
                               success15=float(np.mean([r["success15"] for r in group])),
                               clean_success30=float(np.mean([r["clean_success30"] for r in group])),
                               mean_hold_sec=float(np.mean([r["longest_hold_sec"] for r in group])),
                               mean_finger_peak_n=float(np.mean([r["finger_peak_n"] for r in group])),
                               forbidden_rate=float(np.mean([r["forbidden_rate"] for r in group])),
                               max_penetration_mm=max(r["max_penetration_mm"] for r in group)))
    paired = []
    for scenario in ("nominal", "offset4mm"):
        for method, reference in (("guided","ppo"), ("residual","ppo"), ("bc","ppo"),
                                  ("guided","bc"), ("residual","zero_reference")):
            a = {r["training_seed"]: r for r in runs if r["method"]==method and r["scenario"]==scenario}
            b = {r["training_seed"]: r for r in runs if r["method"]==reference and r["scenario"]==scenario}
            fixed_reference = reference == "zero_reference"
            if fixed_reference:
                if len(b) != 1:
                    raise ValueError("Zero controller must be evaluated once per scenario")
                zero = next(iter(b.values()))
                b = {seed:zero for seed in a}
            if set(a) != set(b) or len(a) < 3:
                raise ValueError("Incomplete paired training seeds")
            delta = np.array([a[s]["success30"]-b[s]["success30"] for s in sorted(a)])
            signs = np.array(list(itertools.product((-1,1), repeat=len(delta))))
            pvalue = float(np.mean(np.abs((signs*delta).mean(axis=1)) >= abs(delta.mean())-1e-12))
            paired.append(dict(method=method, reference=reference, scenario=scenario,
                               training_seed_pairs=len(delta), success_difference=float(delta.mean()),
                               paired_seed_bootstrap95=cluster_interval(delta),
                               exact_seed_signflip_p=None if fixed_reference else pvalue,
                               reference_is_fixed_control=fixed_reference, seed_differences=delta.tolist()))
    for directory in sorted((root / "train").iterdir()):
        if not directory.is_dir():
            continue
        meta = json.loads((directory / "run_config.json").read_text())
        with (directory / "metrics.csv").open() as stream:
            curve = list(csv.DictReader(stream))
        for row in curve:
            learning.append(dict(method=meta["method"], training_seed=meta["seed"],
                                  step=int(row["step"]), success30=float(row["success30"]),
                                  mean_hold=float(row["mean_hold"]), force=float(row["mean_finger_peak_n"])))
    if len(episodes) != protocol["total_test_episodes"]:
        raise ValueError("Incomplete episode count")
    curve_summary = []
    for method, seed in sorted({(r["method"], r["training_seed"]) for r in learning}):
        rows = sorted([r for r in learning if r["method"]==method and r["training_seed"]==seed], key=lambda r:r["step"])
        steps, rates = np.array([r["step"] for r in rows]), np.array([r["success30"] for r in rows])
        if method != "bc" and steps[-1] != protocol["main_ppo_steps"]:
            raise ValueError("Incomplete training budget")
        curve_summary.append(dict(method=method, training_seed=seed, first=float(rates[0]), last=float(rates[-1]),
                                  max=float(rates.max()), best_step=int(steps[rates.argmax()]),
                                  normalized_auc=float(np.trapezoid(rates, steps)/steps[-1]) if steps[-1] else None,
                                  first_dev80_step=int(steps[np.flatnonzero(rates>=.8)[0]]) if np.any(rates>=.8) else None))
    write_csv(output / "episodes.csv", episodes)
    write_csv(output / "per_seed.csv", runs)
    write_csv(output / "learning_curves.csv", learning)
    write_csv(output / "paired.csv", paired)
    write_csv(output / "aggregate.csv", aggregates)
    write_csv(output / "learning_summary.csv", curve_summary)
    failures = []
    for method, scenario in sorted({(r["method"], r["scenario"]) for r in episodes}):
        group = [r for r in episodes if (r["method"],r["scenario"])==(method,scenario)]
        counts = {name:sum(r["outcome"]==name for r in group) for name in
                  ("success", "never_acquired", "insufficient_continuous_hold", "lost_final_validity")}
        failures.append(dict(method=method,scenario=scenario,episodes=len(group), **counts))
    write_csv(output / "outcomes.csv", failures)
    report = dict(status="complete", evaluations=len(runs), episodes=len(episodes), aggregates=aggregates,
                  paired=paired, learning_summary=curve_summary,
                  limitations="Exploratory source/task tuning; three independent training seeds. Seed bootstrap intervals are unstable with n=3; do not treat episode count as independent trained policies. No held-out videos or hardware generalization claim.")
    (output / "statistics.json").write_text(json.dumps(report, indent=2)+"\n")
    lines = ["# 开放起手方块抓握：完整比较统计", "", "## 独立初态结果", "",
             "| 方法 | 初态范围 | 成功/回合 | 30秒成功率 | 各训练种子成功率 | 平均最长保持秒 | 逐指峰值均值N | 最大穿透mm |",
             "|---|---|---|---|---|---|---|---|"]
    labels = dict(ppo="纯PPO", guided="BC后PPO", residual="视频基底+残差PPO", bc="仅BC", zero_reference="仅参考/零修正")
    for r in aggregates:
        rates = ", ".join(f"{100*x:.1f}%" for x in r["seed_rates"])
        lines.append(f"| {labels[r['method']]} | {r['scenario']} | {r['successes']}/{r['episodes']} | {100*r['success30']:.1f}% | {rates} | {r['mean_hold_sec']:.2f} | {r['mean_finger_peak_n']:.2f} | {r['max_penetration_mm']:.3f} |")
    lines += ["", f"共 {len(runs)} 个独立评估批次、{len(episodes)} 条回合记录。训练种子是统计重复单位；不是把全部回合当成独立模型。",
              "", "主指标是无支撑、拇指+其他手指接触、无禁接触、物体未下滑超过10mm，连续保持>=30秒且40秒结束仍有效。",
              "额外的 clean_success30 要求整个撤支撑后回合无禁接触、保存状态穿透<=3mm；原始成功与更严格清洁成功分开给出。",
              "力是仿真逐指法向合力峰值，不是硬件安全认证；手物穿透是保存的20Hz活跃状态检查，不涵盖每个5ms子步或被关闭的自碰撞。",
              "", "## 差值与不确定性", "",
              "配对差值、按训练种子重采样的95%区间和精确符号翻转检验在 paired.csv/statistics.json。只有3个训练种子，区间不稳定，双侧检验最小p值为0.25，不能宣称统计显著。",
              "这是针对单一视频/固定手腕任务的探索性比较，不是跨视频泛化或真机实验证明。", "",
              "## 原始统计", "", "- episodes.csv：每一回合的成功、保持、接触力、非法接触与穿透。",
              "- per_seed.csv：每一个模型/测试范围的统计。", "- learning_curves.csv：全部开发检查点，未只取最佳点。",
              "- aggregate.csv、paired.csv、statistics.json：汇总、配对差值和不确定性。", ""]
    (output / "results.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,4.5))
    for method in ("ppo", "guided", "residual"):
        rows = [r for r in learning if r["method"] == method]
        steps = sorted({r["step"] for r in rows})
        means = [np.mean([r["success30"] for r in rows if r["step"]==s]) for s in steps]
        low = [min(r["success30"] for r in rows if r["step"]==s) for s in steps]
        high = [max(r["success30"] for r in rows if r["step"]==s) for s in steps]
        ax.plot(steps, means, marker="o", label=method)
        ax.fill_between(steps, low, high, alpha=.15)
    ax.set(xlabel="PPO environment steps", ylabel="Development success30", ylim=(-.03,1.03),
           title="Mean across 3 training seeds; shading = seed range, not CI")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
