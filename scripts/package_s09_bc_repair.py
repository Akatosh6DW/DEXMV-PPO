"""Create a small, verifiable report folder; never copy weights or make a zip."""

import argparse
import csv
import json
from pathlib import Path
import shutil

from retarget_s09_reference import digest

ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair"
LABELS = {"old_bc_ppo":"旧BC+PPO", "new_bc":"修复后的BC（未接PPO）",
          "new_bc_ppo":"修复后的BC+PPO", "reference":"校正参考回放"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirmation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--videos", type=Path)
    a = p.parse_args()
    statistics = json.loads((a.confirmation/"report/statistics.json").read_text())
    assert statistics["initial_states_exactly_paired"]
    a.output.mkdir(parents=True, exist_ok=False)
    provenance = []

    def copy(src, relative):
        dest = a.output/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        provenance.append(dict(file=str(relative), source=str(src.resolve()), sha256=digest(dest), bytes=dest.stat().st_size))

    for name in ("BC_PPO修复说明.md", "BC修复迭代.md"):
        copy(ROOT/"docs/experiments/S09"/name, Path(name))
    for src in (a.confirmation/"report").iterdir():
        if src.is_file():
            copy(src, Path("保留测试")/src.name)
    copy(a.confirmation/"plan.json", Path("保留测试/plan.json"))
    plan = json.loads((a.confirmation/"plan.json").read_text())
    for job in plan["jobs"]:
        source = ROOT/job["output"]
        for src in source.rglob("*"):
            if src.is_file() and src.suffix in (".json", ".csv", ".jsonl") and "sources" not in src.parts:
                copy(src, Path("保留测试/逐模型原始记录")/job["name"]/src.relative_to(source))
        copy(source/"step_000000000/initial_states.npz",
             Path("保留测试/逐模型原始记录")/job["name"]/"initial_states.npz")
    cost = dict(recorded_online_transitions=0, completed_bc_updates=0,
                recorded_evaluation_active_transitions=0, development_runs=0,
                limitation="Recorded counts only; interrupted/unlogged work and source-reference construction excluded")
    for run in sorted(REPAIR.glob("I*")):
        if not run.is_dir():
            continue
        cost["development_runs"] += 1
        if (run/"metrics.csv").exists():
            with (run/"metrics.csv").open() as f:
                rows = list(csv.DictReader(f))
            cost["recorded_online_transitions"] += int(float(rows[-1]["step"])) if rows else 0
        if (run/"bc_loss.json").exists():
            rows = json.loads((run/"bc_loss.json").read_text())
            cost["completed_bc_updates"] += int(rows[-1]["step"]) if rows else 0
        for src in run.rglob("*"):
            if src.is_file() and src.suffix in (".json", ".jsonl", ".csv") and "sources" not in src.parts:
                copy(src, Path("开发原始记录")/run.name/src.relative_to(run))
                if src.name == "episodes.json":
                    cost["recorded_evaluation_active_transitions"] += sum(round(e["duration_sec"]/.05)
                        for e in json.loads(src.read_text()))
    for src in (REPAIR/"report").iterdir():
        if src.suffix in (".json", ".csv", ".png"):
            copy(src, Path("开发汇总")/src.name)
    (a.output/"开发汇总/cost.json").write_text(json.dumps(cost, indent=2)+"\n")
    for src in (ROOT/"scripts").glob("*s09*bc*.py"):
        copy(src, Path("关键源码")/src.name)
    for name in ("run_s09_job.py", "orca_s09_env.py", "s09_suspension.py", "train_orca_matched.py"):
        copy(ROOT/"scripts"/name, Path("关键源码")/name)
    for name in ("protocol.json", "demonstrations/dataset_manifest.json"):
        copy(REPAIR.parent/name, Path("历史基线")/name)
    for src in (REPAIR.parent/"report").iterdir():
        if src.is_file() and src.suffix in (".json", ".csv", ".md", ".png"):
            copy(src, Path("历史基线/report")/src.name)
    if a.videos:
        for src in a.videos.iterdir():
            if src.is_file() and src.suffix in (".mp4", ".md", ".json", ".jpg"):
                copy(src, Path("视频")/src.name)
    lines = ["# S09 BC+PPO修复汇报资料", "",
        "这是新修复实验，不覆盖原S09。文件夹用于阅读和移动；没有ZIP、没有权重，也不是独立训练安装包。", "",
        "## 阅读顺序", "",
        "1. [修复说明](BC_PPO修复说明.md)：任务、数据、原理、为什么崩、怎么改、与残差有什么区别。",
        "2. [最终保留测试](保留测试/results.md)：新旧模型在相同初态上的成绩、区间、边界。",
        "3. [逐轮记录](BC修复迭代.md)：所有开发尝试，包含失败，不只展示成功方案。",
        "4. `开发汇总/` 和 `开发原始记录/`：学习曲线CSV、配置、逐回合数据、BC/PPO损失和成本。",
        "5. `保留测试/逐模型原始记录/`：全部测试回合、初始状态NPZ、物理重放审计。",
        "6. `关键源码/`：有注释的关键入口；完整依赖仍在主工程，不能单独复制这些文件直接训练。",
        "7. `历史基线/`：旧S09固定协议与原成绩，避免把新旧测试集混在一起。",
        "8. `视频/`：少量手心视角案例；成功剪到稳定4秒，失败剪到首次失去合格抓持/掉落。", "",
        "## 数字怎么读", "", "| 方法 | 新保留集成功率 |", "|---|---:|"]
    for row in statistics["summaries"]:
        lines.append(f"| {LABELS[row['group']]} | {row['success_rate']:.2%} |")
    lines += ["", "以上是模型组均值；详细报告同时列出每个种子。测试初态只有128个，不能把重复模型执行当成更多独立场景。",
        "本轮没有换新视频或重做RGB视觉提取，奖励与物理判据未改；BC预算、观测条件和调参成本增加。",
        "原权重留在主工程；`保留测试/plan.json`记录对应路径和SHA256，不额外复制占空间。",
        "复制来源见provenance.json；文件完整性见manifest.json。", ""]
    (a.output/"README.md").write_text("\n".join(lines))
    (a.output/"provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+"\n")
    manifest = [dict(file=str(f.relative_to(a.output)), bytes=f.stat().st_size, sha256=digest(f))
                for f in sorted(a.output.rglob("*")) if f.is_file()]
    assert not any(Path(r["file"]).suffix in (".pkl", ".zip") for r in manifest)
    (a.output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    for item in manifest:
        assert digest(a.output/item["file"]) == item["sha256"]
    print(json.dumps(dict(output=str(a.output), files=len(manifest), bytes=sum(r["bytes"] for r in manifest),
                         cost=cost), ensure_ascii=False))


if __name__ == "__main__":
    main()
