"""Post-hoc diagnostics of already consumed confirmation trajectories.

This neither evaluates a new candidate nor estimates stochastic training-state
frequencies. The original frozen confirmation outputs remain untouched.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"


def main():
    source = BASE/"confirmation_20261019/report/episodes.csv"
    with source.open() as stream:
        episodes = list(csv.DictReader(stream))
    rows = []
    for group in sorted({row["group"] for row in episodes}):
        selected = [row for row in episodes if row["group"] == group]
        total = np.array([float(row["total_valid_grasp_sec"]) for row in selected])
        best = np.array([float(row["hold_sec"]) for row in selected])
        assert np.isfinite(total).all() and np.isfinite(best).all()
        assert np.all(best <= total + 5e-6) and np.all(total <= 40)
        # Both counters are multiples of the 0.05-second control period;
        # round away only float32 representation error, not real short holds.
        repeated = np.rint((total-best)/.05)*.05
        rows.append(dict(group=group, executions=len(selected),
            executions_with_recovered_hold=int(np.count_nonzero(repeated > 0)),
            mean_repeated_valid_sec=float(repeated.mean()), max_repeated_valid_sec=float(repeated.max()),
            fraction_of_undiscounted_main_reward=float(repeated.sum()/total.sum())))
    report = dict(scope="Post-hoc on consumed deterministic confirmation cases, not fresh evaluation or training rollouts",
        input_path=str(source.relative_to(ROOT)), input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows)
    output = BASE/"report"
    (output/"hold_recurrence.json").write_text(json.dumps(report, indent=2)+"\n")
    lines = ["# 重复短段是否是主要问题", "",
        "这是已经使用过的150000系列确认轨迹的事后诊断，不是新策略的未见测试，也不是训练随机动作的统计。没有改写原确认报告。", "",
        "原主奖励按全部合格时间给分。R6只按最长连续合格时间给分；两者未折扣主奖励之差对应下表的重复短段时间。其他奖励及折扣仍会影响实际训练。", "",
        "| 组别 | 执行数 | 有重复短段的执行数 | 平均重复秒数 | 占未折扣主奖励 |",
        "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['group']} | {row['executions']} | {row['executions_with_recovered_hold']} | {row['mean_repeated_valid_sec']:.4f} | {row['fraction_of_undiscounted_main_reward']:.3%} |")
    lines += ["", "bc_ppo是开发阶段预选模型；bc_ppo_last是真正训练末步。重复短段确实存在，但在这些执行里只占主奖励的小部分，不能据此认定它是退化的主要原因。", "",
              "E74的R6开发短训末步23/64，低于自身BC26/64和原奖励控制24/64；未支持继续扩大该配方。完整原始统计与输入哈希见同目录hold_recurrence.json。"]
    (output/"hold_recurrence.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
