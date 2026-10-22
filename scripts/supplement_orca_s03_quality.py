"""Descriptive quality sensitivity, separate from S03's frozen primary tests."""

import csv
import json
from pathlib import Path

import numpy as np

from report_orca_suite import write_csv
from run_orca_s03_confirmation import OUT, digest


def boolean(value):
    if value not in ("True", "False"):
        raise ValueError("Expected a serialized boolean, not a missing score")
    return value == "True"


def describe(rows):
    raw = np.array([boolean(r["success30"]) for r in rows])
    forbidden = np.array([boolean(r["substep_forbidden"]) for r in rows])
    clean = np.array([boolean(r["clean_success30"]) for r in rows])
    depth = np.array([float(r["post_release_penetration_m"]) for r in rows])
    force = np.array([float(r["substep_force_peak_n"]) for r in rows])
    if not len(rows) or not np.isfinite(depth).all() or not np.isfinite(force).all():
        raise ValueError("Missing/nonfinite observations")
    np.testing.assert_array_equal(clean, raw & ~forbidden & (depth <= .003))
    curves = [dict(tolerance_mm=t, success_rate=float(np.mean(raw & ~forbidden & (depth <= t/1000))),
                   primary_threshold=t == 3) for t in (2, 3, 4, 5)]
    quality = dict(episodes=len(rows), raw_success_count=int(raw.sum()), clean_success_count=int(clean.sum()),
                   raw_success_rate=float(raw.mean()), clean_success_rate=float(clean.mean()),
                   post_penetration_median_mm=float(np.median(depth)*1000),
                   post_penetration_p95_mm=float(np.quantile(depth, .95)*1000),
                   post_penetration_max_mm=float(depth.max()*1000),
                   force_all_mean_n=float(force.mean()),
                   force_raw_success_mean_n=float(force[raw].mean()) if raw.any() else None,
                   force_clean_success_mean_n=float(force[clean].mean()) if clean.any() else None)
    return curves, quality


def main():
    report = OUT / "report"
    statistics = json.loads((report / "statistics.json").read_text())
    if not statistics["complete"] or statistics["episodes"] != 153600:
        raise ValueError("Quality supplement requires the complete original matrix")
    groups = {}
    with (report / "episodes.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = tuple(row[k] for k in ("condition", "group", "budget", "mode"))
            groups.setdefault(key, []).append(row)
    threshold_rows, quality_rows = [], []
    for key, rows in sorted(groups.items()):
        identifiers = dict(zip(("condition", "group", "budget", "mode"), key))
        curves, quality = describe(rows)
        threshold_rows.extend(dict(**identifiers, **row) for row in curves)
        quality_rows.append(dict(**identifiers, **quality))
    write_csv(report / "quality_thresholds.csv", threshold_rows)
    write_csv(report / "quality_continuous.csv", quality_rows)
    lines = ["# 接触质量的描述性敏感性分析", "",
             "此补充因开发训练中出现‘原始100%、清洁0%但仅略超3mm’而增加，不属于三项预定主检验。",
             "不改变官方3mm阈值、不重新训练、不增加显著性检验；完整列出2/3/4/5mm，不挑最有利阈值。",
             "每一列仍要求原始30秒成功且撤托后无禁接触，只改变穿透上限。", "",
             "## 标准条件长预算", "", "| 方法 | 2mm | 3mm（主指标） | 4mm | 5mm |",
             "|---|---|---|---|---|"]
    nominal = [r for r in quality_rows if r["condition"] == "B0_nominal" and r["budget"] in ("0", "655360")]
    for row in nominal:
        selected = [r for r in threshold_rows if all(r[k] == row[k] for k in ("condition", "group", "budget", "mode"))]
        lines.append("| " + row["group"] + " | " + " | ".join(f"{100*r['success_rate']:.2f}%" for r in selected) + " |")
    lines += ["", "## 连续量与条件分母", "",
              "quality_continuous.csv报告撤托后穿透中位数、95分位和最大值，以及全部回合、原始成功回合、清洁成功回合的指力峰值均值。",
              "条件均值必须同时看对应成功回合数，不能把失败早停带来的低平均力当成控制更优。",
              "质量阈值附近的排名变化需要如实解释；模拟穿透或指力本身都不是现实硬件安全认证。",
              "quality_thresholds.csv覆盖所有条件与预算；不合并共享的BC/零参考控制为额外独立样本。"]
    (report / "quality_sensitivity.md").write_text("\n".join(lines)+"\n")
    manifest = dict(complete=True, groups=len(groups), thresholds_mm=[2, 3, 4, 5], primary_threshold_mm=3,
                    no_new_hypothesis_tests=True, source_sha256=digest(Path(__file__)),
                    original_statistics_sha256=digest(report / "statistics.json"),
                    original_episodes_sha256=digest(report / "episodes.csv"))
    (report / "quality_sensitivity.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
