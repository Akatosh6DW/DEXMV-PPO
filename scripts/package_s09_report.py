"""Collect S09 report material without policy weights or redundant train traces."""

import json
from pathlib import Path
import shutil

from export_s09_showcase import DELIVERY
from retarget_s09_reference import digest
from run_s09_study import OUT, ROOT, check_protocol


def main():
    check_protocol()
    report = OUT / "report"
    audit = json.loads((report / "matrix_verification.json").read_text())
    assert audit["status"] == "matrix_verified" and audit["episodes"] == 2560
    assert (report / "quality_diagnostics.md").exists()
    assert (DELIVERY / "主要视频索引.json").exists()
    copied = []

    def copy(source, relative):
        target = DELIVERY / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if source.suffix == ".md" and source.is_relative_to(ROOT / "docs/experiments/S09"):
            content = target.read_text()
            for old, new in (
                ("../../../artifacts/orca_benchmark_suite/S09/report/", "../01_结果与原始数据/"),
                ("../../../artifacts/orca_benchmark_suite/S09/reward_iteration/report/", "../08_奖励迭代/report/"),
                ("../../../deliverables/S09_多朝向悬吊圆柱_2026-10-15/", "../"),
            ):
                content = content.replace(old, new)
            target.write_text(content)
        copied.append(dict(file=str(relative), bytes=target.stat().st_size,
            sha256=digest(target), source=str(source.relative_to(ROOT)), source_sha256=digest(source),
            relative_links_rebased=digest(target) != digest(source)))

    for source in sorted(report.iterdir()):
        if source.is_file():
            copy(source, Path("01_结果与原始数据") / source.name)
    for source in sorted((ROOT / "docs/experiments/S09").glob("*.md")):
        copy(source, Path("02_思路方案与术语") / source.name)
    copy(OUT / "protocol.json", Path("03_协议与配置/protocol.json"))
    for source in sorted((OUT / "train").glob("*/*")):
        if source.is_file() and source.name in ("metrics.csv", "summary.json", "run_config.json", "bc_loss.json", "ppo_metrics.jsonl"):
            copy(source, Path("04_训练原始记录") / source.relative_to(OUT / "train"))
    for source in sorted((OUT / "eval").glob("*/*/*")):
        if source.is_file() and source.suffix in (".json", ".csv"):
            copy(source, Path("05_逐任务评估记录") / source.relative_to(OUT / "eval"))
    for source in sorted((OUT / "eval").glob("*/*/step_000000000/*")):
        if source.name in ("episodes.json", "initial_states.npz", "native_replay_audit.json"):
            copy(source, Path("05_逐任务评估记录") / source.relative_to(OUT / "eval"))
    for stage in ("P01_source_ik", "P02_bounded_correction", "P03_long_selection", "inputs"):
        for source in sorted((OUT / stage).rglob("*.json")):
            copy(source, Path("06_参考准备记录") / stage / source.relative_to(OUT / stage))
    for name in ("dataset_manifest.json", "summary.json", "metrics.csv", "run_config.json"):
        copy(OUT / "demonstrations" / name, Path("06_参考准备记录/demonstrations") / name)
    copy(OUT / "inputs/reference.npz", Path("06_参考准备记录/inputs/reference.npz"))
    scripts = ("s09_suspension.py", "orca_s09_env.py", "retarget_s09_reference.py",
        "refine_s09_reference.py", "select_s09_long_reference.py", "build_s09_bank.py",
        "run_s09_job.py", "run_s09_study.py", "audit_s09_replay.py",
        "analyze_s09_study.py", "analyze_s09_orientation.py", "summarize_s09_diagnostics.py", "render_s09_trace.py",
        "export_s09_showcase.py", "queue_s09_evaluation.py", "package_s09_report.py", "verify_s09_delivery.py")
    scripts += ("test_s09_suspension.py", "test_s09_env.py", "test_s09_reward.py", "test_s09_audit.py")
    if (OUT / "reward_iteration/report/verification.json").exists():
        scripts += ("run_s09_reward_job.py", "run_s09_reward_study.py", "analyze_s09_reward_study.py")
    for name in scripts:
        copy(ROOT / "scripts" / name, Path("07_核心源码阅读副本") / name)
    for source in sorted((OUT / "reward_iteration").rglob("*")):
        if source.is_file() and source.suffix in (".md", ".json", ".csv", ".jsonl", ".png"):
            copy(source, Path("08_奖励迭代") / source.relative_to(OUT / "reward_iteration"))
    (DELIVERY / "文件清单.json").write_text(json.dumps(copied, ensure_ascii=False, indent=2) + "\n")
    lines = ["# S09汇报资料总入口", "", "本文件夹可整体移动，不需要ZIP；不包含策略权重和全量训练轨迹。", "",
        "## 建议阅读顺序", "",
        "1. [用人话解释结果](02_思路方案与术语/结果解读.md)：做成了什么、没做成什么，如何理解差异。",
        "2. [原理与完整链路](02_思路方案与术语/原理与链路说明.md)：视频到底提供什么、四种方法如何出动作。",
        "3. [固定四组实验结果](01_结果与原始数据/results.md)：统计和图表。",
        "4. [接触质量与失败分析](01_结果与原始数据/quality_diagnostics.md)：穿透、力峰值及回放核验。",
        "5. [主要视频目录](目录.md)：手心角度，外观与碰撞几何。",
        "6. [逐轮日志](02_思路方案与术语/iteration_log.md)：过程、失败与后续决策。", "",
        "## 文件内容", "",
        "- `01_结果与原始数据`：2560回合CSV、逐种子曲线、统计区间、配对差值、图表、离线成本及审计。",
        "- `02_思路方案与术语`：中文说明、协议与逐轮记录。",
        "- `03_协议与配置`：冻结实验协议及源码/模型/数据哈希。",
        "- `04_训练原始记录`：9次训练的监测、配置、BC误差和PPO日志，无权重。",
        "- `05_逐任务评估记录`：50项评估的逐回合结果、初态和保存预览的回放审计。",
        "- `06_参考准备记录`：IK、物理校正、参考选择、示范采集记录及6模板参考库。",
        "- `07_核心源码阅读副本`：关键算法实现，便于讲解；不是可脱离原工程独立运行的完整软件包。",
        "- `08_奖励迭代`：仅在实际执行后包含相应分支记录，不混入固定R0结果。", "",
        "原始完整工程、场景网格、示范集及权重仍保留在主工程`artifacts/orca_benchmark_suite/S09/`，本文件夹未删除或移动它们。配置内绝对路径用于追溯原机，不表示移动文件夹后可以直接复训。",
        "本轮使用已有DexMV处理后示范，不声称重新从RGB提取；当前控制策略读取仿真物体姿态和模拟触觉，不是实时视觉端到端网络。",
        "复制文件及原始来源/哈希见`文件清单.json`；视频各子目录另有render.json哈希。"]
    if (OUT / "reward_iteration/report/verification.json").exists():
        lines += ["", "## 单独的奖励迭代", "",
            "[同奖励续训与新增穿透惩罚的探索性对比](08_奖励迭代/report/results.md)。这不是主实验R0的替代成绩。"]
    (DELIVERY / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(output=str(DELIVERY), files=len(copied), copied_bytes=sum(r["bytes"] for r in copied)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
