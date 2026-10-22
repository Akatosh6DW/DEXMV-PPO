"""Summarize matched perturbation diagnostics without claiming held-out scores."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from retarget_s09_reference import digest


def paired_rows(rows):
    indexed = {(r["run"], r["seed"], r["impulse"], r["replay"]): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate diagnostic records")
    pairs = []
    for row in rows:
        if row["replay"] or row["impulse"] == "none":
            continue
        other = indexed[(row["run"], row["seed"], row["impulse"], True)]
        assert row["nominal_alive_at_end"] == other["nominal_alive_at_end"]
        result = dict(run=row["run"], seed=row["seed"], impulse=row["impulse"],
            nominal_full_horizon=row["nominal_alive_at_end"],
            feedback_success=row["success30"], replay_success=other["success30"],
            feedback_hold_sec=row["hold_sec"], replay_hold_sec=other["hold_sec"],
            feedback_depth_mm=row["max_released_penetration_m"]*1000,
            replay_depth_mm=other["max_released_penetration_m"]*1000)
        for horizon in (1, 4):
            key = f"position_deviation_{horizon}s_m"
            a, b = row[key], other[key]
            result[f"feedback_deviation_{horizon}s_mm"] = None if a is None else a*1000
            result[f"replay_deviation_{horizon}s_mm"] = None if b is None else b*1000
            result[f"feedback_minus_replay_{horizon}s_mm"] = None if a is None or b is None else (a-b)*1000
        pairs.append(result)
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = args.input
    read = lambda p: json.loads(p.read_text())
    summary, protocol, rows = [read(root/n) for n in ("summary.json", "protocol.json", "results.json")]
    assert summary["status"] == "completed" and not summary["smoke"]
    assert summary["cases"] == len(rows) == 160 and protocol["suffix_steps"] == 760
    pairs = paired_rows(rows)
    for row in rows:
        file = root/row["run"]/f"seed{row['seed']}_{row['impulse']}_{'replay' if row['replay'] else 'feedback'}.npz"
        assert digest(file) == row["trace_sha256"]
        assert row["max_qpos_error"] < 3e-6 and row["max_qvel_error"] < 1e-4
        assert row["released_rope_force_n"] == 0
    overview = []
    for run in dict.fromkeys(r["run"] for r in rows):
        all_pairs = [r for r in pairs if r["run"] == run]
        full = [r for r in all_pairs if r["nominal_full_horizon"]]
        row = dict(run=run, perturbation_pairs=len(all_pairs), nominal_full_horizon_pairs=len(full),
            feedback_successes_full_nominal=sum(r["feedback_success"] for r in full),
            replay_successes_full_nominal=sum(r["replay_success"] for r in full),
            feedback_gained_full_nominal=sum(r["feedback_success"] and not r["replay_success"] for r in full),
            feedback_lost_full_nominal=sum(not r["feedback_success"] and r["replay_success"] for r in full))
        for horizon in (1,4):
            values = [r[f"feedback_minus_replay_{horizon}s_mm"] for r in all_pairs
                      if r[f"feedback_minus_replay_{horizon}s_mm"] is not None]
            row.update({f"paired_alive_{horizon}s":len(values),
                f"median_feedback_minus_replay_{horizon}s_mm":float(np.median(values)) if values else None,
                f"feedback_worse_over_1um_{horizon}s":sum(x>.001 for x in values),
                f"feedback_better_over_1um_{horizon}s":sum(x<-.001 for x in values)})
        overview.append(row)
    out = root/"report"
    out.mkdir(exist_ok=True)
    for name, data in (("pairs.csv",pairs),("overview.csv",overview)):
        with (out/name).open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    lines = ["# 同状态小扰动：反馈还是动作本身不稳", "",
        "四个已知训练初态，共同I27前2秒；不是未见成功率，也不是可部署观测回放。",
        "只在名义轨迹活到40秒的案例统计整段反馈得失。其他案例仍保留，仅比较共同活动时段；禁止把回放冻结终止观测的影响算成反馈收益。",
        "位置偏差是相对同一策略无扰动轨迹的距离，不等于离目标的距离。更小偏差不自动代表更会抓。正负1微米用于描述差异，不是统计显著性门槛。", "",
        "| 模型 | 名义完整的扰动对数 | 反馈成功 | 回放成功 | 4秒偏差差值中位数mm |",
        "|---|---:|---:|---:|---:|"]
    for row in overview:
        value = row["median_feedback_minus_replay_4s_mm"]
        lines.append(f"| {row['run']} | {row['nominal_full_horizon_pairs']} | {row['feedback_successes_full_nominal']} | {row['replay_successes_full_nominal']} | {value} |")
    lines += ["", "`pairs.csv`保留全部64个扰动对及接触质量；`overview.csv`含1/4秒共同存活数和方向计数。",
        "无扰动两路径逐元素一致、阶段型负对照共同活动帧逐元素一致，执行脚本中已断言；全部160条后缀均独立原生回放。",
        "成功/失败最多只来自四个预选训练案例，不能做总体成功率或泛化宣称。此报告不改变原任务标准或选取最终模型。"]
    (out/"results.md").write_text("\n".join(lines)+"\n")
    (out/"evidence.json").write_text(json.dumps(dict(status="completed_diagnostic_only",
        results_sha256=digest(root/"results.json"), protocol_sha256=digest(root/"protocol.json"),
        summary_sha256=digest(root/"summary.json"), source_sha256=digest(Path(__file__)),
        rows=overview), indent=2)+"\n")
    print(json.dumps(overview))


if __name__ == "__main__":
    main()
