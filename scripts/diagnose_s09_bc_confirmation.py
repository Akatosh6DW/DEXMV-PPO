"""Post-hoc pose coverage diagnostics, without changing tested candidates."""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirmation", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads((a.confirmation/"plan.json").read_text())
    stats = json.loads((a.confirmation/"report/statistics.json").read_text())
    rows = []
    for job in plan["jobs"]:
        # The original pure-interface BC does not encode this remembered mode.
        if job["group"] == "old_bc_ppo":
            continue
        es = json.loads((ROOT/job["output"]/"step_000000000/episodes.json").read_text())
        for mode in range(6):
            cases = [e for e in es if e["selected_reference"] == mode]
            rows.append(dict(group=job["group"], training_seed=job["training_seed"], mode=mode,
                successes=sum(e["success30"] for e in cases), attempts=len(cases),
                excessive_penetration_cases=sum(e["max_released_penetration_m"] > .003 for e in cases)))
    out = a.confirmation/"report"
    with (out/"pose_coverage.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    transitions = stats["transitions"]
    lost, gained = sum(r["lost"] for r in transitions), sum(r["gained"] for r in transitions)
    lines = ["# 保留测试后的瓶颈诊断", "",
        "这是已冻结测试之后的描述性分析，不用于修改本轮权重或重新选择模型。", "",
        f"新BC接PPO后，在三个模型的384个配对执行中，失去{lost}个原有成功，新增{gained}个成功。",
        "新增与失去会相互抵消，所以仅看平均成功率会遗漏个别场景被破坏的问题。", "",
        "## 初始方向模式", "",
        "模式是按初始圆柱轴与六个参考轴的接近程度划分，不等于六个互不重叠的自然抓取任务。",
        "不同模式样本数不同；表中分母是实际测试数，不能把同一初态的三个模型执行视为三个新初态。", "",
        "| 模式 | 新BC | 新BC+PPO | 参考回放 |", "|---|---:|---:|---:|"]
    for mode in range(6):
        values = []
        for group in ("new_bc", "new_bc_ppo", "reference"):
            subset = [r for r in rows if r["group"] == group and r["mode"] == mode]
            s, n = sum(r["successes"] for r in subset), sum(r["attempts"] for r in subset)
            values.append(f"{s}/{n}")
        lines.append(f"| {mode} | " + " | ".join(values) + " |")
    lines += ["", "本批模式3出现明确遗忘：新BC原有14/66次成功，接PPO后变为0/66；不能因为总体比旧版强就忽略这类系统性退化。",
        "这个分组是测试后的诊断，不是在这批测试上继续挑参数的授权。", "", "## 接触质量", "",
        "| 方法 | 全部回合最大释放后穿透mm | 全部回合峰值力N | 成功回合峰值力N |",
        "|---|---:|---:|---:|"]
    for r in stats["summaries"]:
        force = r["successful_case_peak_force_n"]
        force_text = f"{force:.2f}" if force is not None else "无成功回合"
        lines.append(f"| {r['group']} | {r['all_case_max_penetration_mm']:.3f} | "
            f"{r['all_case_peak_force_n']:.2f} | {force_text} |")
    lines += ["", "超过3mm的失败回合仍留在总分母里。成功判据不约束峰值力，不能将上表解释为硬件安全认证。", "",
        "## 下一步真正要解决的问题", "",
        "1. 覆盖薄弱方向：采集这些方向的真实可行示范和恢复动作，而不是只重复已有保持段。原BC训练模式4没有成功示范。",
        "2. 保护已学会的成功：本轮试过的离线约束和在线KL在旧BC起点上未带来稳定增益；不能凭名称就认定它们解决了遗忘。",
        "3. 提高反馈收益：当前BC学的是模式/时间规律，PPO反馈权重虽确实更新，但还没有在统计上证明净收益。低噪声保护已有动作，也可能限制新抓法探索。",
        "4. 新视频实验单独做：保留来源、手物标定、重定向误差及失败轨迹，不能把旋转旧参考叫作新视频。",
        "5. 若继续改奖励或训练配方，需要新的保留测试集；本次120000--120127已经查看，不能再次称其未参与开发。", "",
        "以上是后续方向，不是本轮已经完成的实验。本轮已验证大幅改善旧版成功率，尚未解决所有朝向与PPO净增益。"]
    (out/"bottlenecks.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(lost=lost, gained=gained, output=str(out)), ensure_ascii=False))


if __name__ == "__main__":
    main()
