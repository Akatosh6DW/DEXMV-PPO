"""Summarize completed/partial repair runs without selecting on held-out tests."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/orca_benchmark_suite/S09/bc_repair"


def main():
    output = OUT / "report"
    output.mkdir(exist_ok=True)
    runs, curves = [], []
    for path in sorted(OUT.glob("I*")):
        if not (path/"metrics.csv").exists():
            continue
        with (path/"metrics.csv").open() as file:
            rows = [{k: float(v) for k,v in row.items()} for row in csv.DictReader(file)]
        if not rows:
            continue
        config = json.loads((path/"run_config.json").read_text())
        # Match each runner's summary: first occurrence of maximum success.
        best = max(rows, key=lambda r: r["success30"])
        bc = json.loads((path/"bc_loss.json").read_text()) if (path/"bc_loss.json").exists() else []
        item = dict(run=path.name, complete=(path/"summary.json").exists(),
            seed=config["seed"], development_episodes=config["eval_count"],
            first=rows[0]["success30"], last=rows[-1]["success30"], max=best["success30"],
            best_step=int(best["step"]), last_step=int(rows[-1]["step"]),
            hold_first=rows[0]["hold_sec"], hold_last=rows[-1]["hold_sec"],
            hold_max=max(r["hold_sec"] for r in rows),
            hold_best_step=int(max(rows, key=lambda r: r["hold_sec"])["step"]),
            final_depth_mm=rows[-1]["max_penetration_m"]*1000,
            final_peak_force_n=rows[-1]["peak_force_n"], bc_updates=bc[-1]["step"] if bc else 0,
            checkpoint_parent=config.get("checkpoint"), learning_rate=config["learning_rate"],
            observation_variant=config["env_config"].get("observation_variant", "see repair_config.json"),
            path=str(path.relative_to(ROOT)))
        runs.append(item)
        curves.extend([dict(run=path.name, **row) for row in rows])
    for name, data in (("iterations", runs), ("learning_curves", curves)):
        (output/(name+".json")).write_text(json.dumps(data, indent=2)+"\n")
        if data:
            with (output/(name+".csv")).open("w") as file:
                writer = csv.DictWriter(file, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    lines = ["# BC+PPO修复开发记录", "",
        "这是反复调参使用的开发集，不是最终独立测试成绩；16例和32例的结果不能直接混为同一个评估集合。",
        "结果按实际文件读取，未完成的训练明确标为partial。失败分支未删除。", "",
        "| 迭代 | 状态 | 开发例数 | BC更新 | 在线已记录步数 | first | last | max | best step | 最终穿透mm |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        lines.append(f"| {r['run']} | {'complete' if r['complete'] else 'partial'} | {r['development_episodes']} | "
            f"{r['bc_updates']} | {r['last_step']} | {r['first']:.2%} | {r['last']:.2%} | "
            f"{r['max']:.2%} | {r['best_step']} | {r['final_depth_mm']:.3f} |")
    lines += ["", "max用于描述曲线，不自动替代预先选定的最终检查点。所有成功仍要求释放后严格30秒和结束时合格。",
        "增加BC预算/观测条件/新数据的分支必须分开说明，不能称为只改学习率。",
        "[逐轮原因与操作日志](../../../../../docs/experiments/S09/BC修复迭代.md)"]
    (output/"development.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
