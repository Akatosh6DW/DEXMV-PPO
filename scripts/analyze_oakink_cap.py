"""Aggregate the complete cap comparison by training seed, including failures."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oakink_cap_protocol import SEEDS, recipes
from orca_formal_statistics import paired_effect, seed_estimate
from prepare_hic_orca_object import digest
from run_oakink_cap_evaluations import jobs
from oakink_cap_failure_diagnostics import from_file as failure_diagnostics


def analyze(suite, roots, output, allow_partial=False):
    expected = {job["label"]: job for job in jobs()}
    specifications = {recipe["label"]: recipe for recipe in recipes()}
    reports, failure_stages = {}, {}
    for root in roots:
        for path in root.glob("*/report.json"):
            name = path.parent.name
            if name not in expected or name in reports:
                raise ValueError("Unexpected or duplicate cap evaluation: " + name)
            report = json.loads(path.read_text())
            job = expected[name]
            if (Path(report["run"]).resolve() != (suite / job["run"]).resolve()
                    or report["scenario"] != job["scenario"] or report["baseline"] != job["baseline"]
                    or report["development"] or report["seeds"] != list(range(28500, 28564))):
                raise ValueError("Wrong cap report identity/population")
            checked = json.loads((path.parent / "independent_audit.json").read_text())
            if not checked["passed"] or checked["report_sha256"] != digest(path):
                raise ValueError("Missing or stale cap independent audit")
            if digest(path.parent / "trajectories.npz") != report["trace_sha256"]:
                raise ValueError("Cap trace changed")
            spec = specifications[job["run"]]
            assert report["train_seed"] == spec["seed"] and report["method"] == spec["method"]
            reports[name] = report
            failure_stages[name] = failure_diagnostics(path.parent / "trajectories.npz")
    missing = sorted(expected.keys() - reports.keys())
    if missing and not allow_partial:
        raise ValueError(f"Cap results incomplete: {len(missing)}/110 missing")
    grouped, baselines = defaultdict(dict), {}
    for name, report in reports.items():
        if report["baseline"]:
            baselines[name] = report["metrics"]
        else:
            grouped[(report["method"], report["scenario"])][report["train_seed"]] = report["metrics"]
    table = []
    metric_names = ("success", "release_rate", "mean_max_degrees", "mean_final_degrees", "mean_best_hold_s",
                    "peak_force_n", "max_penetration_mm", "forbidden_fraction", "terminal_failure_rate")
    for (method, scenario), results in sorted(grouped.items()):
        if not allow_partial and set(results) != set(SEEDS):
            raise ValueError("Incomplete cap training-seed group")
        table.append(dict(method=method, scenario=scenario, per_seed=results,
                          metrics={name: seed_estimate([results[s][name] for s in sorted(results)]) for name in metric_names}))
    effects = []
    for left, right in (("residual", "pure"), ("guided", "pure"), ("guided", "bc"), ("residual", "guided")):
        a, b = grouped.get((left, "nominal"), {}), grouped.get((right, "nominal"), {})
        if set(a) == set(b) == set(SEEDS):
            effects.append(dict(left=left, right=right, success=paired_effect(
                {s: a[s]["success"] for s in SEEDS}, {s: b[s]["success"] for s in SEEDS})))
    output.mkdir(parents=True, exist_ok=True)
    result = dict(evaluations=len(reports), expected_evaluations=110, evaluations_complete=not missing,
                  entire_goal_complete=False, missing=missing, table=table, paired_effects=effects,
                  zero_controls=baselines, failure_diagnostics=failure_stages)
    (output / "statistics.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# 同源视频单手拧盖：正式统计", "", "独立评估：" + f"{len(reports)}/110。",
             "状态：" + ("独立测试齐全，仍需结合视觉和全项目审计。" if not missing else "未完成，不作最终排名。"), "",
             "主成功为完整脱离并在末帧连续有效持盖一秒，不是旋转几十度。仅适用于当前近似螺旋机构。",
             "每个策略先在 64 初态上评分，再按五个训练种子统计；置信区间不以初态数量代替训练次数。", "",
             "| 方法 | 成功率均值/标准差 | 95% 区间 | 螺旋约束解除率 | 最大转角均值 | 峰值力均值 N |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in table:
        if row["scenario"] != "nominal":
            continue
        m = row["metrics"]
        s = m["success"]
        std = "NA" if s["std"] is None else f"{s['std']:.1%}"
        lines.append(f"| {row['method']} | {s['mean']:.1%}/{std} | [{s['bootstrap95'][0]:.1%}, {s['bootstrap95'][1]:.1%}] | "
                     f"{m['release_rate']['mean']:.1%} | {m['mean_max_degrees']['mean']:.2f} deg | {m['peak_force_n']['mean']:.2f} |")
    lines += ["", "## 解释边界", "", "螺旋约束解除率（release_rate）只表示曾解除约束，不表示瓶盖已移离瓶身或稳定持盖。",
              "全部零成功也是有效负结果，不能改成功阈值或丢弃失败种子。",
              "零宽 bootstrap 区间不代表总体确定；仅五种子时双侧精确符号翻转检验最小 p=0.0625。",
              "源视频关节点不是本次从 RGB 重新估计，Orca 固定底座不能复制人体手臂运输。",
              "视频零修正和固定起手的原始成绩、质量/摩擦敏感性及逐种子数据见 statistics.json。",
              "几何/质量/摩擦和解析螺旋约束的假设见冻结协议，不外推为真实硬件性能。"]
    (output / "results.md").write_text("\n".join(lines) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for method in ("bc", "pure", "guided", "residual"):
        values = []
        for seed in SEEDS:
            run = suite / f"s{seed}_{method}"
            if not (run / "formal_audit.json").exists():
                continue
            with (run / "metrics.csv").open() as stream:
                values.append([{k: float(v) for k, v in r.items()} for r in csv.DictReader(stream)])
        if not values:
            continue
        steps = [r["step"] for r in values[0]]
        for axis, metric in zip(axes, ("success", "mean_max_degrees")):
            samples = np.array([[r[metric] for r in rows] for rows in values])
            mean = samples.mean(axis=0)
            sd = samples.std(axis=0, ddof=1) if len(samples) > 1 else np.zeros_like(mean)
            axis.plot(steps, mean, marker="o" if method == "bc" else None, label=f"{method} n={len(values)}")
            axis.fill_between(steps, mean - sd, mean + sd, alpha=.12)
            axis.set_xlabel("PPO environment steps")
            axis.grid(alpha=.2)
    axes[0].set_ylabel("Complete cap removal and final hold: success")
    axes[0].set_ylim(-.03, 1.03)
    axes[1].set_ylabel("Maximum rotation (degrees), not success")
    axes[1].legend()
    figure.suptitle("Development monitoring, not held-out test scores")
    figure.savefig(output / "development_curves.png", dpi=160)
    plt.close(figure)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--evaluations", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    report = analyze(args.suite, args.evaluations, args.output, args.allow_partial)
    print("CAP EVALUATIONS", report["evaluations"], "/110", flush=True)


if __name__ == "__main__":
    main()
