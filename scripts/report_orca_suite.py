"""Recompute S02 tables with crossed seed/case bootstrap and paired contrasts."""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = dict(ppo="纯PPO", guided="BC后PPO", bc="仅BC", residual="完整残差",
             linear="线性合拢+残差", unrefined="未校正参考+残差",
             zero_video="校正参考/零修正", zero_linear="线性合拢/零修正",
             zero_unrefined="未校正参考/零修正", half_residual="完整残差/修正减半")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_interval(values, seed=711, draws=5000):
    """Crossed axes: training seeds and shared test cases, never fake zero runs."""
    values = np.asarray(values, float)
    if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
        raise ValueError("Expected finite seed x case matrix")
    rng = np.random.default_rng(seed)
    nr, nc = values.shape
    rw = rng.multinomial(nr, np.full(nr, 1/nr), size=draws) / nr
    cw = rng.multinomial(nc, np.full(nc, 1/nc), size=draws) / nc
    sample = np.sum((rw @ values) * cw, axis=1)
    return np.quantile(sample, [.025, .975]).tolist()


def sign_flip_pvalue(difference):
    difference = np.asarray(difference, float)
    if difference.ndim != 1 or not 0 < len(difference) <= 20 or not np.isfinite(difference).all():
        raise ValueError("Expected <=20 paired training seed differences")
    signs = np.array(list(itertools.product((-1, 1), repeat=len(difference))))
    null = np.abs(np.mean(signs * difference, axis=1))
    return float(np.mean(null >= abs(difference.mean()) - 1e-12))


def holm(values):
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Expected finite p-values in [0, 1]")
    order = np.argsort(values)
    output = np.empty(len(values))
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, (len(values)-rank)*values[index])
        output[index] = min(1., running)
    return output.tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    plan_path = root / "statistics_plan.json"
    plan = json.loads(plan_path.read_text())
    output = root / "report"
    output.mkdir(exist_ok=True)
    per_seed, episodes, curves = [], [], []
    grouped = {}
    missing = []
    for record in protocol["jobs"]:
        directory = root / record["name"]
        if not (directory / "summary.json").exists():
            missing.append(record["name"])
            continue
        job = json.loads((ROOT / record["path"]).read_text())
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["status"] == "completed" and summary["job_sha256"] == record["sha256"]
        if record["name"].startswith("train/"):
            with (directory / "metrics.csv").open() as stream:
                for row in csv.DictReader(stream):
                    curves.append(dict(method=job["method"], seed=job["seed"], step=int(row["step"]),
                                       success30=float(row["success30"]), mean_hold=float(row["mean_hold"]),
                                       elapsed_sec=float(row["elapsed_sec"])))
            continue
        label = record["name"].split("/")[-1]
        method = label.rsplit("_s", 1)[0] if "_s" in label else label
        benchmark = job["benchmark"]
        data = json.loads((directory / "step_000000000/episodes.json").read_text())
        if [item["seed"] for item in data] != protocol["test_seeds"]:
            raise ValueError("Test population/order mismatch")
        if any(item["clean_success30"] is None for item in data):
            raise ValueError("Final evaluation requires native substep audit")
        grouped.setdefault((benchmark, method), []).append((job["seed"], data))
        for item in data:
            episodes.append(dict(benchmark=benchmark, method=method, train_seed=job["seed"], **item))
        per_seed.append(dict(benchmark=benchmark, method=method, seed=job["seed"], episodes=len(data),
                             success30=np.mean([r["success30"] for r in data]),
                             clean_success30=np.mean([r["clean_success30"] for r in data]),
                             mean_hold=np.mean([r["longest_hold_sec"] for r in data]),
                             mean_peak_force_n=np.mean([r["substep_force_peak_n"] for r in data]),
                             max_penetration_m=max(r["max_penetration_m"] for r in data)))
    if missing and not args.allow_partial:
        raise ValueError(f"Matrix incomplete: {len(missing)} jobs; use --allow-partial for explicitly partial tables")
    aggregate, matrices = [], {}
    for (benchmark, method), runs in sorted(grouped.items()):
        runs.sort(key=lambda item: item[0])
        expected = [1001] if method.startswith("zero_") else protocol["train_seeds"]
        if [r[0] for r in runs] != expected:
            continue
        flat = [row for _, rows in runs for row in rows]
        clean = np.array([[r["clean_success30"] for r in rows] for _, rows in runs], float)
        raw = np.array([[r["success30"] for r in rows] for _, rows in runs], float)
        matrices[(benchmark, method)] = clean
        ci = bootstrap_interval(clean)
        pre = [r for r in flat if r["valid_before_push"]]
        aggregate.append(dict(benchmark=benchmark, method=method, policies=len(runs), episodes=len(flat),
                              clean_success30=float(clean.mean()), raw_success30=float(raw.mean()),
                              ci_low=ci[0], ci_high=ci[1],
                              mean_hold=float(np.mean([r["longest_hold_sec"] for r in flat])),
                              mean_peak_force_n=float(np.mean([r["substep_force_peak_n"] for r in flat])),
                              max_penetration_mm=1000*max(r["max_penetration_m"] for r in flat),
                              pre_push_valid=len(pre),
                              post_push_valid_given_pre=float(np.mean([r["valid_one_second_after_push"] for r in pre])) if pre else None))
    contrasts = []
    for benchmark in protocol["benchmark_names"]:
        a = matrices.get((benchmark, "residual"))
        for control in ("ppo", "guided", "zero_video", "linear", "unrefined"):
            b = matrices.get((benchmark, control))
            if a is None or b is None:
                continue
            difference = a-b
            by_seed = difference.mean(axis=1)
            low, high = bootstrap_interval(difference)
            contrasts.append(dict(benchmark=benchmark, comparison="residual-"+control,
                                  difference_pp=float(difference.mean()*100),
                                  ci_low_pp=low*100, ci_high_pp=high*100,
                                  seed_conditional_p=sign_flip_pvalue(by_seed),
                                  positive_seeds=int(np.sum(by_seed > 0)),
                                  zero_seeds=int(np.sum(by_seed == 0)),
                                  negative_seeds=int(np.sum(by_seed < 0))))
    complete_family = len(contrasts) == 30
    corrected = holm([r["seed_conditional_p"] for r in contrasts]) if complete_family else [None]*len(contrasts)
    for row, corrected_p in zip(contrasts, corrected):
        row["holm_30_p"] = corrected_p
        row["primary_holm_5_p"] = None
        row["supports_primary_positive_difference"] = False
        row["supports_global_positive_difference"] = bool(corrected_p is not None and corrected_p < .05
                                                          and row["ci_low_pp"] > 0)
    primary = [r for r in contrasts if r["benchmark"] == plan["primary_benchmark"]]
    if len(primary) == len(plan["primary_controls"]):
        for row, pvalue in zip(primary, holm([r["seed_conditional_p"] for r in primary])):
            row["primary_holm_5_p"] = pvalue
            row["supports_primary_positive_difference"] = bool(pvalue < .05 and row["ci_low_pp"] > 0)
    write_csv(output / "episodes.csv", episodes)
    write_csv(output / "per_seed.csv", per_seed)
    write_csv(output / "aggregate.csv", aggregate)
    write_csv(output / "paired.csv", contrasts)
    write_csv(output / "learning_curves.csv", curves)
    statistics = dict(complete=not missing, missing=missing, aggregate=aggregate, contrasts=contrasts,
                      evaluator="crossed bootstrap 5000 draws over training seeds and shared initial states",
                      hypothesis_test=plan["test"], statistics_plan=plan,
                      statistics_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                      zero_control="one controller evaluated once; broadcasting only forms paired differences, not extra independent samples",
                      protocol_sha256=hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest(),
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / "statistics.json").write_text(json.dumps(statistics, indent=2)+"\n")
    lines = ["# S02 方块任务族与消融统计", "",
             f"状态：{'完成' if not missing else '未完成，以下仅为已完成部分'}；缺失运行 {len(missing)}；已统计测试回合 {len(episodes)}。", "",
             "主指标是每5 ms求解接触审计的撤托后清洁30秒成功；同时保留旧20 Hz原始成功标准。",
             "最大穿透列覆盖整段活跃回合，主清洁判据只检查撤托后；两者时间范围不同，不能混读。",
             "各方法的准备成本不同，训练配方继承M01。所有模型使用最终检查点，不按测试选优。", "",
             "| 条件 | 方法 | 独立策略/回合 | 清洁成功率[95%区间] | 原始成功率 | 最长保持均值s | 子步指力峰值均值N | 最大穿透mm |",
             "|---|---|---|---|---|---|---|---|"]
    for row in aggregate:
        lines.append(f"| {row['benchmark']} | {NAMES[row['method']]} | {row['policies']}/{row['episodes']} | "
                     f"{100*row['clean_success30']:.2f}% [{100*row['ci_low']:.2f}, {100*row['ci_high']:.2f}] | "
                     f"{100*row['raw_success30']:.2f}% | {row['mean_hold']:.2f} | "
                     f"{row['mean_peak_force_n']:.2f} | {row['max_penetration_mm']:.3f} |")
    lines += ["", "## 消融与差值", "",
              "主假设为B0的五项比较，Holm-5校正；另完整报告全部30项的Holm-30。区间是逐项区间，不是同时置信区间。",
              "10个训练种子的双侧精确检验最小p=0.001953125，因此Holm-30不可能达到0.05；不能宣称所有条件均有统计显著优势。", "",
              "| 条件 | 比较 | 差值百分点 | 95%区间 | 主假设Holm-5 p | 全局Holm-30 p | 正/平/负训练种子 |", "|---|---|---|---|---|---|---|"]
    for row in contrasts:
        pvalue = "待全矩阵完成" if row["holm_30_p"] is None else f"{row['holm_30_p']:.4f}"
        primary_p = ("不适用（扩展条件）" if row["benchmark"] != plan["primary_benchmark"] else
                     "主比较未齐" if row["primary_holm_5_p"] is None else f"{row['primary_holm_5_p']:.4f}")
        lines.append(f"| {row['benchmark']} | {row['comparison']} | {row['difference_pp']:+.2f} | "
                     f"[{row['ci_low_pp']:+.2f}, {row['ci_high_pp']:+.2f}] | {primary_p} | {pvalue} | "
                     f"{row['positive_seeds']}/{row['zero_seeds']}/{row['negative_seeds']} |")
    lines += ["", "## 解读边界", "",
              "- 不支持正向差值不等于证明两者相等；零修正没有10个训练种子，不重复伪造。",
              "- B1..B5是同一方块的零再训练迁移，不是新物体/新视频实验。",
              "- 轨迹消融保留视频来源的末姿态，不能单靠它证明视频优于人工优化抓型。",
              "- B5恢复率需同时看扰动前有效抓握的分母；完整无条件成功率仍为主结果。",
              "- 自碰撞未启用，物体状态来自仿真，力与穿透不代表硬件安全认证。",
              "- 逐回合episodes.csv、逐种子per_seed.csv、全部开发learning_curves.csv可独立处理。"]
    (output / "results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(complete=not missing, missing=len(missing), episodes=len(episodes), output=str(output))))


if __name__ == "__main__":
    main()
