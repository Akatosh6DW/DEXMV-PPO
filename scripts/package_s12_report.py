"""Collect completed S12 evidence without duplicating weights or large trajectories."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--videos", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = a.study / "report"
    statistics = json.loads((report / "statistics.json").read_text())
    verification = json.loads((report / "verification.json").read_text())
    if statistics["status"] != "completed" or verification["data_integrity"] != "passed":
        raise ValueError("Only completed, audited data may be packaged")
    if verification["total_episodes"] != 1560:
        raise ValueError("Unexpected final evaluation count")
    video_manifest = json.loads((a.videos / "视频来源.json").read_text())
    if len(video_manifest) != 10 or not (a.videos / "目录.md").exists():
        raise ValueError("Incomplete main video set")
    for item in video_manifest:
        if hashlib.sha256((a.videos / item["file"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Video checksum mismatch")
    conclusion = ROOT / "docs/experiments/S12/实验总结.md"
    if not conclusion.exists():
        raise ValueError("The human-readable final analysis must be written first")
    a.output.mkdir(parents=True, exist_ok=False)
    copied = []

    def copy(source, relative):
        target = a.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(dict(file=str(relative), source=str(source.resolve()),
            bytes=target.stat().st_size, sha256=hashlib.sha256(target.read_bytes()).hexdigest()))

    for source in sorted((ROOT / "docs/experiments/S12").glob("*.md")):
        copy(source, Path("说明") / source.name)
    for source in sorted(report.iterdir()):
        if source.is_file() and source.suffix in (".csv", ".json", ".md", ".png"):
            copy(source, Path("数据与图表") / source.name)
    for name in ("protocol.json", "evaluation_plan.json", "environment.json",
                 "environment_after_restart.json", "environment_evaluation_recovery_20261024.json",
                 "dataset_audit.json"):
        copy(a.study / name, Path("协议与来源") / name)
    plan = json.loads((a.study / "protocol.json").read_text())
    for job in plan["jobs"]:
        run = Path(job["output"])
        for name in ("run_config.json", "metrics.csv", "summary.json"):
            copy(run / name, Path("训练原始记录") / job["name"] / name)
        if (run / "bc_loss.json").exists():
            copy(run / "bc_loss.json", Path("训练原始记录") / job["name"] / "bc_loss.json")
    evaluation = json.loads((a.study / "evaluation_plan.json").read_text())
    for job in evaluation["jobs"]:
        run = Path(job["output"])
        for name in ("evaluation_config.json", "episodes.json", "summary.json"):
            copy(run / name, Path("评估原始记录") / job["name"] / name)
    for folder in sorted(a.study.parent.glob("V2_*")):
        if folder.is_dir():
            for source in sorted(folder.rglob("*")):
                if source.is_file() and source.suffix in (".json", ".csv"):
                    copy(source, Path("开发验证记录") / folder.name / source.relative_to(folder))
    for source in sorted(a.videos.iterdir()):
        if source.is_file():
            copy(source, Path("手心视频") / source.name)
    for name in ("s12_soft.py", "orca_s12_env.py", "run_s12_job.py",
                 "train_orca_matched.py", "train_orca_comparison.py", "s12_physics_worker.py",
                 "evaluate_s12_checkpoint.py", "analyze_s12_study.py",
                 "diagnose_s12_failures.py", "verify_s12_study.py"):
        copy(ROOT / "scripts" / name, Path("核心源码快照") / name)
    manifest = dict(study=str(a.study.resolve()), data_integrity=verification["data_integrity"],
        physics_status=verification["status"], total_evaluation_episodes=1560,
        files=copied, total_copied_bytes=sum(row["bytes"] for row in copied),
        exclusions=["策略权重", "完整 NPZ 轨迹", "网格资产", "旧 V1 作废开发结果"])
    (a.output / "文件清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    lines = ["# S12 软体圆柱汇报资料", "",
        "本文件夹是可整体移动的汇报材料，不是脱离原工程即可训练的安装包；没有重复复制策略权重。",
        "", "## 建议阅读顺序", "",
        "1. [实验总结](说明/实验总结.md)：做了什么、实际结果、原因、结论边界。",
        "2. [设计与原理](说明/设计与原理.md)：视频参考、BC、PPO、残差和软体物理。",
        "3. [正式结果表](数据与图表/results.md)及[失败分析](数据与图表/failure_analysis.md)。",
        "4. [手心视频目录](手心视频/目录.md)：同一初态的五种控制器，各有外观/碰撞体两版。",
        "5. [数据字段说明](说明/数据字段说明.md)和[冻结方案](说明/正式对比协议.md)。",
        "", "## 包含与不包含", "",
        "- 数据与图表：全部 1560 个独立评估回合的汇总行、各种子统计、置信区间、失败诊断和图表。",
        "- 训练原始记录：9 组训练的 CSV、配置和结果摘要。",
        "- 评估原始记录：65 组评估的原始逐回合 JSON、配置和摘要；不是只挑成功案例。",
        "- 开发验证记录：V2 材料/摆位/步长验证、示范筛选、短训练、A01/A02 等诊断的 JSON/CSV。它们不进入正式 1560 回合分母；作废 V1 不混入。",
        "- 协议与来源：冻结哈希、数据多样性审计和运行环境。",
        "- 核心源码快照：供阅读的关键实现；完整依赖和模型仍在原工程，不宣称是独立安装包。",
        "- 不含大型 NPZ、权重和模型资产。表中的 trace/source_run 为原机器绝对路径，移动本文件夹不会携带这些文件。",
        f"- 原始完整研究目录：`{a.study.resolve()}`。",
        "", "说明文档保留原工程相对链接；移出工程后，请优先使用本目录的上述本地链接和随包图表。",
        "`文件清单.json` 给出每个复制文件的原始位置、大小和 SHA256。物理模型失败与数据完整性分别记录，数据完整不代表所有回合的材料模拟都有效。"]
    (a.output / "目录.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(files=len(copied), bytes=manifest["total_copied_bytes"],
                         output=str(a.output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
