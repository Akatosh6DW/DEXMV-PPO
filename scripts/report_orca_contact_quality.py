"""Descriptive all-phase penetration supplement; preserve the S02 primary metric."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    groups, missing, hashes = {}, [], {}
    for job in protocol["jobs"]:
        if not job["name"].startswith("eval/"):
            continue
        directory = root / job["name"]
        if not (directory / "summary.json").exists():
            missing.append(job["name"])
            continue
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["job_sha256"] == job["sha256"] and summary["status"] == "completed"
        benchmark, label = job["name"].split("/")[1:]
        method = label.rsplit("_s", 1)[0] if "_s" in label else label
        path = directory / "step_000000000/episodes.json"
        hashes[job["name"]] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = json.loads(path.read_text())
        assert [r["seed"] for r in rows] == protocol["test_seeds"]
        groups.setdefault((benchmark, method), []).append(rows)
    records = []
    for (benchmark, method), runs in sorted(groups.items()):
        expected = 1 if method.startswith("zero_") else len(protocol["train_seeds"])
        if len(runs) != expected:
            continue
        data = [r for run in runs for r in run]
        raw = sum(r["success30"] for r in data)
        primary = sum(r["clean_success30"] for r in data)
        whole = sum(r["clean_success30"] and r["max_penetration_m"] <= .003 for r in data)
        pre_only = sum(r["max_penetration_m"] > .003 and r["post_release_penetration_m"] <= .003 for r in data)
        assert whole <= primary <= raw
        records.append(dict(benchmark=benchmark, method=method, policies=len(runs), episodes=len(data),
                            raw_success30=raw/len(data), predefined_postrelease_clean=primary/len(data),
                            supplemental_all_phase_clean=whole/len(data),
                            primary_successes_failing_pre_release_depth=primary-whole,
                            any_pre_only_penetration_failures=pre_only,
                            max_all_phase_penetration_mm=1000*max(r["max_penetration_m"] for r in data),
                            max_postrelease_penetration_mm=1000*max(r["post_release_penetration_m"] for r in data)))
    report = root / "report"
    report.mkdir(exist_ok=True)
    if records:
        with (report / "contact_quality_supplement.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    result = dict(complete=not missing, missing=missing, groups=records, input_hashes=hashes,
                  primary_metric_unchanged=True, additional_hypothesis_tests=0,
                  criterion="Supplement only: predefined clean_success30 AND all-active-substep max_penetration_m <=0.003; support contacts before release remain permitted",
                  note="M01 used saved-state all-active penetration; S02 primary uses native-substep postrelease penetration. Neither metric should be silently substituted for the other.",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (report / "contact_quality_supplement.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["# 接触质量范围补充", "", f"状态：{'完成' if not missing else '未完成'}。",
             "S02预定主指标只对撤托后的穿透/禁接触做清洁验收，最大穿透列则覆盖整段活跃回合。",
             "所以某组最大穿透超过3mm、但原始和主清洁成功率相同，并不一定是统计错误。",
             "下面补充整段回合手物穿透也不超过3mm的结果；不更换主指标，不增加显著性检验。",
             "M01使用20Hz保存状态的整段活跃穿透，和S02的采样频率、时间范围都不同，不能直接横比。", "",
             "| 条件 | 方法 | 回合 | 预定撤托后清洁成功 | 补充整段穿透合格成功 | 因撤托前深穿透被补充指标排除的成功数 |",
             "|---|---|---|---|---|---|"]
    for row in records:
        lines.append(f"| {row['benchmark']} | {row['method']} | {row['episodes']} | "
                     f"{100*row['predefined_postrelease_clean']:.2f}% | {100*row['supplemental_all_phase_clean']:.2f}% | "
                     f"{row['primary_successes_failing_pre_release_depth']} |")
    lines += ["", "整段穿透验收仍不是手部自碰撞或真机安全认证；撤托前托板接触属于任务允许的支撑。"]
    (report / "contact_quality_supplement.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(complete=not missing, groups=len(records), missing=len(missing))))
    if missing and not args.allow_partial:
        raise RuntimeError("Incomplete evaluation matrix")


if __name__ == "__main__":
    main()
