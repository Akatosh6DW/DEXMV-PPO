"""Aggregate all declared formal evaluations by independent training seed."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orca_formal_protocol import SEEDS, recipes, scenarios
from orca_formal_statistics import paired_effect, seed_estimate
from prepare_hic_orca_object import digest
from run_orca_formal_evaluations import jobs
from audit_orca_formal_evaluation import check_report_identity
from orca_failure_diagnostics import from_file as failure_diagnostics


def learning_diagnostics(suite):
    result = []
    for spec in recipes():
        run = suite / spec["label"]
        if not (run / "formal_audit.json").exists():
            continue
        with (run / "metrics.csv").open() as stream:
            rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        x = np.asarray([row["step"] for row in rows])
        y = np.asarray([row["success30"] for row in rows])
        assert x[0] == 0 and x[-1] == spec["steps"]
        saved = json.loads((run / "summary.json").read_text())
        auc = float(np.trapezoid(y, x) / x[-1]) if x[-1] else float(y[0])
        hit = next((int(x[i - 1]) for i in range(1, len(x)) if min(y[i - 1], y[i]) >= .75), None)
        np.testing.assert_allclose(saved["auc"], auc, atol=1e-10)
        assert saved["sustained75_step"] == hit
        paired_bc = json.loads((suite / spec["bc"] / "summary.json").read_text())["bc_elapsed_sec"] if spec["bc"] else 0.
        result.append(dict(run=spec["label"], phase=spec["phase"], method=spec["method"], seed=spec["seed"],
                           normalized_auc=auc, first=float(y[0]), last=float(y[-1]), maximum=float(y.max()),
                           peak_to_final_regression=float(y.max() - y[-1]), first_two_checkpoints_at_least75_step=hit,
                           threshold_not_reached=hit is None, bc_fit_seconds=saved["bc_elapsed_sec"],
                           paired_bc_fit_seconds=paired_bc,
                           monitoring_elapsed_seconds=rows[-1]["elapsed_sec"], steps=x.tolist(), success=y.tolist()))
    return dict(runs=result, expected_training_jobs=80, complete=len(result) == 80,
                interpretation="Development curves, not held-out performance. The 75% metric means two consecutive recorded checkpoints, not sustained success through the final checkpoint. Null is right-censored, not zero steps. BC fitting has no PPO interaction budget.")


def analyze(suite, roots, output, allow_partial=False):
    specifications = {r["label"]: r for r in recipes()}
    expected_jobs = {j["label"]: j for j in jobs("all")}
    expected = set(expected_jobs)
    reports, failure_stages = {}, {}
    for root in roots:
        for path in root.glob("*/report.json"):
            if path.parent.name not in expected:
                raise ValueError("Unexpected evaluation: " + str(path))
            report = json.loads(path.read_text())
            check_report_identity(report, expected_jobs[path.parent.name], suite)
            audit = json.loads((path.parent / "independent_audit.json").read_text())
            if not audit["passed"] or audit["report_sha256"] != digest(path):
                raise ValueError("Missing or stale independent evaluation audit")
            if report["development_smoke"] or report["seeds"] != list(range(25000, 25064)):
                raise ValueError("Development/incorrect test seeds in formal aggregation")
            if digest(path.parent / "trajectories.npz") != report["trace_sha256"]:
                raise ValueError("Evaluation trace changed")
            if path.parent.name in reports:
                raise ValueError("Duplicate evaluation")
            reports[path.parent.name] = report
            failure_stages[path.parent.name] = failure_diagnostics(path.parent / "trajectories.npz")
    missing = sorted(expected - reports.keys())
    if missing and not allow_partial:
        raise ValueError(f"Formal report incomplete: missing {len(missing)}/{len(expected)} evaluations")
    grouped = defaultdict(dict)
    baselines = {}
    for name, report in reports.items():
        if report["zero_correction_baseline"]:
            baselines[name] = report["metrics"]
            continue
        label = Path(report["run"]).name
        spec = specifications[label]
        assert report["train_seed"] == spec["seed"]
        assert (report["method"], report["tactile"], report["reference_mode"]) == (
            spec["method"], spec["tactile"], spec["reference_mode"])
        key = (label.split("_", 1)[1], report["scenario"])
        grouped[key][spec["seed"]] = report
    table = []
    for (condition, scenario), by_seed in sorted(grouped.items()):
        complete = set(by_seed) == set(SEEDS)
        if not complete and not allow_partial:
            raise ValueError("Incomplete seed group")
        metrics = {}
        for name in ("success30", "strict_mean_hold", "success15", "valid_final", "peak_force_n", "acquisition_rate"):
            metrics[name] = seed_estimate([by_seed[s]["metrics"][name] for s in sorted(by_seed)])
        metrics["max_penetration_mm"] = seed_estimate(
            [1000 * by_seed[s]["geometry"]["max_penetration_m"] for s in sorted(by_seed)])
        table.append(dict(condition=condition, scenario=scenario, complete_seeds=complete, metrics=metrics,
                          invalid_initial_counts={s: by_seed[s]["metrics"]["invalid_initial_count"] for s in sorted(by_seed)}))
    pairs = []
    contrasts = [(f"{m}_on", f"{m}_off", "tactile") for m in ("bc", "ppo", "guided", "residual")]
    contrasts += [(f"{m}_{t}", f"ppo_{t}", "architecture") for t in ("off", "on") for m in ("guided", "residual")]
    contrasts += [(f"residual_{t}", f"residual_{mode}_{t}", "reference")
                  for t in ("off", "on") for mode in ("endpoint", "initial")]
    contrasts += [(f"guided_{t}", f"guided_smallstd_{t}", "exploration") for t in ("off", "on")]
    for left, right, kind in contrasts:
        for scenario in (scenarios() if kind in ("tactile", "architecture") else ["nominal"]):
            a, b = grouped.get((left, scenario), {}), grouped.get((right, scenario), {})
            if set(a) != set(SEEDS) or set(b) != set(SEEDS):
                continue
            effect = paired_effect({s: a[s]["metrics"]["success30"] for s in SEEDS},
                                   {s: b[s]["metrics"]["success30"] for s in SEEDS})
            pairs.append(dict(left=left, right=right, kind=kind, scenario=scenario, effect=effect))
    output.mkdir(parents=True, exist_ok=True)
    data = dict(evaluations_complete=not missing, entire_goal_complete=False,
                evaluations=len(reports), expected_evaluations=len(expected), missing=missing,
                table=table, paired_effects=pairs, zero_correction=baselines,
                learning=learning_diagnostics(suite),
                failure_diagnostics=failure_stages,
                note="Cube-only statistical report; cap task and visual/audit deliverables are separate requirements")
    (output / "statistics.json").write_text(json.dumps(data, indent=2) + "\n")
    lines = ["# 正式方块实验统计", "",
             "状态：" + ("独立评估齐全，尚须结合完整审计和复杂任务报告。" if not missing else "未完成，仅供检查进度，不能作最终排名。"),
             f"已收集 {len(reports)}/{len(expected)} 份预定评估。", "",
             "每个数据点先在固定的 64 个初态上计分，再以独立训练种子为单位统计。区间为种子 bootstrap 95%。",
             "五种子仍然有限，区间为零宽不意味着总体结果确定。最终检查点固定，不根据测试成绩挑选。", "",
             "## 名义条件", "",
             "| 条件 | 训练种子数 | 成功率均值 / 标准差 | 95% 区间 | 严格平均保持秒 | 逐种子峰值力的均值 N |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in table:
        if row["scenario"] != "nominal":
            continue
        m = row["metrics"]
        s = m["success30"]
        std = "NA" if s["std"] is None else f"{s['std']:.1%}"
        lines.append(f"| {row['condition']} | {s['n_training_seeds']} | {s['mean']:.1%} / {std} | "
                     f"[{s['bootstrap95'][0]:.1%}, {s['bootstrap95'][1]:.1%}] | "
                     f"{m['strict_mean_hold']['mean']:.3f} | {m['peak_force_n']['mean']:.2f} |")
    lines += ["", "## 解释边界", "",
              "触觉与架构的配对差值、全部敏感性条件和逐种子数值见 statistics.json。",
              "0% 不能解释为从未接触物体；主标准同时要求保持 30 秒且末帧有效。",
              "质量/摩擦不是实物标定，手部自碰撞仍有局限；结果不证明硬件安全或任意物体泛化。",
              "探索 std、残差权限和示范数据质量均需纳入解释；本主表比较统一配方，不声称各方法已被最佳调参。",
              "学习曲线 AUC、峰值到末尾退化及两个连续检查点达到 75% 的步数见 learning 字段。未达到记为删失，不能记为 0 步。",
              "字段名 sustained75_step 在历史训练摘要中仅表示连续两个记录点过线，不表示后续永不退化。BC 拟合时间另列。",
              "精确双侧符号翻转检验五对样本最小 p=0.0625；不将多项次要测试中有利的结果挑作显著主结论。"]
    (output / "results.md").write_text("\n".join(lines) + "\n")
    plot_learning_curves(suite, output)
    return data


def plot_learning_curves(suite, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colors = dict(bc="#6f6f6f", ppo="#267aad", guided="#ce7132", residual="#25886b")
    for method in ("bc", "ppo", "guided", "residual"):
        for tactile in ("off", "on"):
            curves = []
            for seed in SEEDS:
                path = suite / f"s{seed}_{method}_{tactile}" / "metrics.csv"
                if not (path.parent / "formal_audit.json").exists():
                    continue
                with path.open() as stream:
                    curves.append([{k: float(v) for k, v in r.items()} for r in csv.DictReader(stream)])
            if not curves:
                continue
            x = [r["step"] for r in curves[0]]
            for axis, metric in zip(axes, ("success30", "mean_hold")):
                values = np.array([[r[metric] for r in curve] for curve in curves])
                average = values.mean(axis=0)
                std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(len(x))
                label = f"{method}, touch {tactile} (n={len(curves)})"
                axis.plot(x, average, "-" if tactile == "on" else "--", color=colors[method],
                          marker="o" if method == "bc" else None, label=label)
                axis.fill_between(x, average - std, average + std, color=colors[method], alpha=.08)
    axes[0].set_ylabel("30 s + valid final: success rate")
    axes[1].set_ylabel("Longest valid hold (s)")
    axes[0].set_ylim(-.03, 1.03)
    axes[1].set_ylim(0, 40)
    for axis in axes:
        axis.set_xlabel("PPO environment steps")
        axis.grid(alpha=.2)
    axes[1].legend(fontsize=7, ncol=2, loc="lower right")
    fig.suptitle("Development curves: mean +/- training-seed SD; not held-out test results", fontsize=11)
    fig.savefig(output / "development_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    data = analyze(args.suite, args.evaluations, args.output, args.allow_partial)
    print("EVALUATIONS", data["evaluations"], "/", data["expected_evaluations"], "COMPLETE", data["evaluations_complete"])


if __name__ == "__main__":
    main()
