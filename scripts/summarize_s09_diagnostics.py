"""Supplement the frozen comparison with transparent failure and force summaries."""

import csv
import json

import numpy as np

from analyze_s09_study import CHINESE
from run_s09_study import METHODS, OUT, STRATA, check_protocol


def main():
    check_protocol()
    report = OUT / "report"
    verification = json.loads((report / "matrix_verification.json").read_text())
    assert verification["status"] == "matrix_verified" and verification["episodes"] == 2560
    with (report / "episodes.csv").open() as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    for stratum in STRATA:
        for method in (*METHODS, "reference"):
            selected = [r for r in raw if r["stratum"] == stratum and r["method"] == method]
            success = np.array([r["success30"] == "True" for r in selected])
            numeric = lambda key: np.array([float(r[key]) for r in selected])
            hold, depth, force = map(numeric, ("hold_sec", "max_released_penetration_m", "peak_force_n"))
            rows.append(dict(stratum=stratum, method=method, evaluations=len(selected),
                successes=int(success.sum()), mean_hold_sec=float(hold.mean()),
                median_hold_sec=float(np.median(hold)),
                reached_30s_but_failed=int(((hold >= 30) & ~success).sum()),
                released_penetration_over_3mm=int((depth > .003).sum()),
                initial_penetration_over_1mm=int((numeric("initial_penetration_m") > .001).sum()),
                ended_before_40s=int((numeric("duration_sec") < 39.999).sum()),
                final_invalid=sum(r["final_valid"] != "True" for r in selected),
                forbidden_support=sum(r["forbidden"] == "True" for r in selected),
                max_penetration_mm=float(depth.max() * 1000),
                success_max_penetration_mm=float(depth[success].max() * 1000) if success.any() else None,
                force_median_n=float(np.median(force)), force_p95_n=float(np.quantile(force, .95)),
                force_max_n=float(force.max()),
                success_force_max_n=float(force[success].max()) if success.any() else None,
                residual_rms_mean_rad=float(numeric("residual_rms_rad").mean())))
    (report / "quality_diagnostics.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (report / "quality_diagnostics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audits = []
    for file in sorted((OUT / "eval").glob("*/*/step_000000000/native_replay_audit.json")):
        audit = json.loads(file.read_text())
        assert audit["status"] == "verified" and all(e["passed"] for e in audit["episodes"])
        audits.append(dict(file=str(file.relative_to(OUT)), **audit))
    assert len(audits) == 50, "All 50 evaluation preview batches must be replay-audited"
    (report / "replay_audits.json").write_text(json.dumps(audits, indent=2) + "\n")
    lines = ["# S09接触质量与失败诊断", "",
        "每个学习组的数量包括3个训练种子的评估回合，不代表这么多个独立训练。失败原因可以重叠，不能直接相加。",
        "最大力是每回合所有活动时刻的单个接触法向力峰值，再跨回合统计；不是合力，也不是关节力矩。成功标准未限制力峰值，因此成功不等于硬件安全。", "",
        "| 朝向 | 方法 | 成功/评估次数 | 平均最长保持(s) | 释放后穿透>3mm | 最长保持30s但最终失败 | 力峰值P95/最大(N) |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['stratum']} | {CHINESE[r['method']]} | {r['successes']}/{r['evaluations']} | "
            f"{r['mean_hold_sec']:.2f} | {r['released_penetration_over_3mm']} | {r['reached_30s_but_failed']} | "
            f"{r['force_p95_n']:.2f}/{r['force_max_n']:.2f} |")
    lines += ["", "## 轨迹核验", "",
        f"50项评估共保存并原生回放核验{sum(len(a['episodes']) for a in audits)}条预览轨迹，覆盖每项保存的成功/失败例。",
        "这不是把全部2560个回合重新跑一遍，也不是使用另一套独立物理引擎。核验在RL循环外按保存的控制量执行原生MuJoCo，共享同一套绳力与接触定义，比较逐步位置、速度、接触审计及成功判定。",
        "残差预览额外检查参考＋有界修正的动作合成；纯PPO及BC后PPO检查观测中的参考片段为零。",
        "", "- [完整质量统计](quality_diagnostics.csv)", "- [逐预览回放审计](replay_audits.json)"]
    (report / "quality_diagnostics.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(groups=len(rows), audited_jobs=len(audits))), flush=True)


if __name__ == "__main__":
    main()
