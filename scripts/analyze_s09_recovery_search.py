"""Keep teacher searches separate from learned-policy benchmark statistics."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"
RUNS = ("D16_recovery_teacher_preflight", "D17_bc_recovery_teacher_preflight",
        "D18_bc_failed_state_teacher", "D19_retry_recovery_teacher_longcheck",
        "D20_recovery_teacher_horizon8")


def main():
    hashes, rows, details = {}, [], []
    def read(path):
        raw = path.read_bytes()
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    for name in RUNS:
        path = BASE/name
        summary, protocol, cases = [read(path/file) for file in ("summary.json", "protocol.json", "results.json")]
        assert summary["status"] == "completed" and len(cases) == summary["cases"]
        long = "unchanged_successes" in summary
        field, baseline = ("success30", "unchanged_bc") if long else ("recovery_candidate", "unchanged_reference")
        before = sum(r[baseline][field] for r in cases)
        after = sum(r["selected_offset"][field] for r in cases)
        assert before == summary["unchanged_successes" if long else "unchanged_candidates"]
        assert after == summary["corrected_successes" if long else "selected_candidates"]
        rows.append(dict(run=name, criterion="40s episode / strict 30s hold" if long else "4s local continuation (last 1s valid)",
            n=len(cases), before=before, after=after,
            gained=sum(not r[baseline][field] and r["selected_offset"][field] for r in cases),
            lost=sum(r[baseline][field] and not r["selected_offset"][field] for r in cases),
            native_control_steps=summary.get("total_native_control_steps", "not_recorded_in_early_probe")))
        if long:
            for case in cases:
                details.append(dict(run=name, seed=case["seed"], mode=case["mode"],
                    before=case[baseline][field], after=case["selected_offset"][field],
                    before_hold=case[baseline]["best_hold_sec_including_prefix"],
                    after_hold=case["selected_offset"]["best_hold_sec_including_prefix"],
                    before_duration=case[baseline]["final_time_sec"],
                    after_duration=case["selected_offset"]["final_time_sec"],
                    after_depth_mm=case["selected_offset"]["max_penetration_m"]*1000,
                    after_peak_force=case["selected_offset"]["peak_force_n"]))
    out = BASE/"report/recovery_teacher"
    out.mkdir(parents=True, exist_ok=True)
    for filename, values in (("overview.csv", rows), ("long_cases.csv", details)):
        with (out/filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (out/"evidence.json").write_text(json.dumps(dict(rows=rows, long_cases=details, inputs=hashes,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Teacher diagnostics on training states only. No policy was trained; cohorts and metrics must not be pooled."), indent=2)+"\n")
    lines = ["# 恢复教师前置诊断：短预测是否会误导", "",
        "不是BC/PPO新模型成功率。D16/D17是轻微扰动，D18是42个训练初态筛查；D19/D20只复核其中4个被查询的失败续段。不同分母、不同标准不合并。", "",
        "| 实验 | 条件 | 例数 | 原动作合格 | 搜索后合格 | 新增/丢失 |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['run']} | {row['criterion']} | {row['n']} | {row['before']} | {row['after']} | {row['gained']}/{row['lost']} |")
    lines += ["", "## 全部四个查询案例的长回放", "",
        "| 实验 | 初态 | 原最长合格秒 | 修正后最长合格秒 | 原结束秒 | 修正后结束秒 | 修正后最大穿透mm | 峰值N |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in details:
        lines.append(f"| {r['run']} | {r['seed']} | {r['before_hold']:.2f} | {r['after_hold']:.2f} | {r['before_duration']:.2f} | {r['after_duration']:.2f} | {r['after_depth_mm']:.3f} | {r['after_peak_force']:.3f} |")
    lines += ["", "## 边界", "",
        "- 物理、物体、摩擦及严格30秒成功标准没有修改。局部搜索是在当前状态附近选一个固定目标偏移，不是已经训练好的反馈网络或完整DAgger/DART。",
        "- D19首次读取计数器的类型错误保留在原目录，正式结果来自修复后的retry目录。D20在相同四个状态上使用8秒预测，再做40秒验证，不按验证结果继续选动作。",
        "- D16/D17主要说明原动作已能承受该轻扰动；D18局部改善必须经过长回放，不能用4秒条件冒充30秒成功。",
        "- 搜索消耗的额外仿真属于教师数据成本，不是免费PPO增益。D18/D20已记录总原生控制步，早期D16/D17未单独记录全部重复回放成本。",
        "- 所有案例保留原始搜索JSON与轨迹NPZ，来源与哈希见evidence.json。D16至D19早期主要保存源代码哈希，未全部保留当时的完整源码快照；D20额外保存主要脚本快照。"]
    (out/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
