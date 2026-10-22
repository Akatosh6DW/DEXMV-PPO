"""Render existing reporting CSVs as readable tables; never recompute experiments."""

import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/汇报资料_2026-10-10"
STUDIES = [
    ("方块 S02", "artifacts/orca_benchmark_suite/S02", 77600),
    ("方块 S03（部分结果）", "artifacts/orca_benchmark_suite/S03", 91800),
    ("方块 S04（新初态复测）", "artifacts/orca_benchmark_suite/S04", 4100),
    ("圆柱 S02", "artifacts/orca_cylinder_s02", 77600),
]
METHODS = {
    "ppo": "纯 PPO", "guided": "BC 后 PPO", "residual": "视频参考 + 残差 PPO",
    "bc": "仅 BC", "linear": "线性参考 + 残差", "unrefined": "未校正参考 + 残差",
    "zero_video": "视频参考零修正", "zero": "视频参考零修正", "zero_linear": "线性参考零修正",
    "zero_unrefined": "未校正参考零修正", "half_residual": "执行时修正减半",
    "ppo_fixed": "固定配方 PPO", "ppo_tuned": "开发选定 PPO",
    "guided_tuned": "开发选定 BC 后 PPO", "residual_tuned": "开发选定残差 PPO",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percent(value):
    return f"{100 * float(value):.2f}%"


def table(headers, rows):
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + [
        "| " + " | ".join(map(escape, row)) + " |" for row in rows
    ]


def main():
    lines = ["# 实验完整数据附录", "", "生成日期：2026-10-11。由原始统计 CSV 自动转成 Markdown，不重新训练、不改变评分或检验。",
             "", "[返回实验全解](03_实验全解.md) · [资料总目录](README.md)", "",
             "## 阅读规则", "", "- 比例转换为百分数；百分点差值不再乘 100。",
             "- 区间来自原统计文件，未重新抽样。显示仅作小数舍入。",
             "- `policies` 是策略数，`episodes` 是策略与初态组合数，不等于独立训练数。",
             "- S03 未完成整体矩阵；空缺 p 值显示未报告，不能解读为零或显著。",
             "- S04 只有 B0 新初态复测；其他条件应查 S03，不冒充 S04 新结果。",
             "- 所有原字段在链接的 CSV 中，本文不把不同实验合成一个总成功率。", ""]
    audit = []
    total_aggregate = 0
    for label, rel, expected in STUDIES:
        paths = {}
        for name in ("aggregate.csv", "episodes.csv", "per_seed.csv", "paired.csv", "learning_curves.csv"):
            source = ROOT / rel / "report" / name
            copied = OUT / rel / "report" / name
            if sha(source) != sha(copied):
                raise ValueError("Reporting copy differs: " + str(copied))
            paths[name] = copied
            audit.append(dict(path=str(copied.relative_to(OUT)), sha256=sha(copied)))
        with paths["episodes.csv"].open() as stream:
            episode_count = sum(1 for _ in csv.DictReader(stream))
        assert episode_count == expected, (label, episode_count)
        with paths["aggregate.csv"].open() as stream:
            aggregate = list(csv.DictReader(stream))
        total_aggregate += len(aggregate)
        lines += [f"## {label}", "", f"逐回合表现有 **{episode_count} 行**。",
                  " · ".join(f"[{name}]({rel}/report/{name})" for name in paths), "",
                  "### 全部已报告汇总", ""]
        rows = []
        for r in aggregate:
            method = r.get("method", r.get("group"))
            interval = f"[{percent(r['ci_low'])}, {percent(r['ci_high'])}]"
            rows.append([r.get("benchmark", r.get("condition", "B0 新初态")),
                         METHODS.get(method, method), r.get("budget", "163840" if method not in
                         ("bc", "zero", "zero_video", "zero_linear", "zero_unrefined") else "0"),
                         r["policies"], r["episodes"], percent(r.get("clean_success30", r.get("clean"))),
                         percent(r.get("raw_success30", r.get("raw"))), interval])
        lines += table(["条件", "方法", "在线步数", "策略数", "回合数", "清洁成功率", "原始成功率", "清洁率 95% 区间"], rows)
        if "S03" in rel:
            lines += ["", "固定 PPO 与开发选定 PPO 的最终配方相同，同种子组用于重复性核验，不是两个独立改进算法。"]
        lines += ["", "### 已保存的配对比较", ""]
        with paths["paired.csv"].open() as stream:
            paired = list(csv.DictReader(stream))
        rows = []
        for r in paired:
            p = r.get("primary_holm_5_p", r.get("primary_holm_3_p", r.get("holm_2_p", "")))
            support = r.get("supports_primary_positive_difference", r.get("supported_on_this_benchmark", ""))
            rows.append([r.get("benchmark", ""), r["comparison"], f"{float(r['difference_pp']):+.2f}",
                         f"[{float(r['ci_low_pp']):+.2f}, {float(r['ci_high_pp']):+.2f}]",
                         p or "未报告/不适用", r.get("holm_30_p", "不适用"), support or "未报告"])
        lines += table(["条件", "比较（前者减后者）", "差值/百分点", "95% 差值区间/百分点", "主比较校正 p", "Holm-30 p", "主比较支持正向收益"], rows)
        lines += ["", "支持列按原 CSV 保留；False 不等于证明两个方法相同。S03 扩展比较不是已完成的主检验。", ""]
    lines += ["## 文件追溯", "", "本附录逐一核验 20 份 CSV 与原项目文件的 SHA256 一致。",
              "各实验的每种子统计、学习曲线、逐回合数据均由上述入口打开；没有将其伪装成新的独立样本。",
              "图表、机制解释和字段词典见[实验全解](03_实验全解.md)。", ""]
    target = OUT / "04_实验完整数据附录.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    documents = []
    links_checked = 0
    for path in sorted(OUT.glob("*.md")):
        documents.append(dict(path=path.name, bytes=path.stat().st_size, sha256=sha(path)))
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" in link:
                continue
            relative, _, anchor = link.partition("#")
            linked = path.parent / relative if relative else path
            if not linked.exists():
                raise FileNotFoundError(f"{path.name}: {link}")
            if anchor and f'id="{anchor}"' not in linked.read_text():
                raise ValueError(f"Missing explicit section anchor: {link}")
            links_checked += 1
    index = OUT / "indexes/documents.csv"
    with index.open(encoding="utf-8-sig") as stream:
        existing = list(csv.DictReader(stream))
    updated = {r["path"] for r in documents}
    with index.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows([r for r in existing if r["path"] not in updated] + documents)
    report = dict(date="2026-10-11", verified_csv=audit, aggregate_rows=total_aggregate,
                  episode_rows=sum(s[2] for s in STUDIES), generated_path=target.name,
                  generated_sha256=sha(target), documents=documents,
                  document_links_checked=links_checked, original_data_modified=False)
    (OUT / "indexes/full_explanation_data_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(csv_verified=len(audit), aggregate_rows=total_aggregate,
                          episode_rows=report["episode_rows"], document=str(target)), ensure_ascii=False))


if __name__ == "__main__":
    main()
