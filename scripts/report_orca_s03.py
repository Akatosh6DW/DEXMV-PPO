"""Recompute S03 statistics without selecting policies on final test results."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from report_orca_suite import bootstrap_interval, sign_flip_pvalue, holm, write_csv
from run_orca_s03_confirmation import OUT, ROOT, SEEDS, digest

NAMES = dict(ppo_fixed="固定PPO", ppo_tuned="调优PPO", guided_tuned="调优BC后PPO",
             residual_tuned="调优残差", bc="仅BC", zero="零修正参考")


def contrast(a, b, name, primary=False):
    delta = np.asarray(a, float) - np.asarray(b, float)
    if delta.ndim != 2 or delta.shape[0] != 10:
        raise ValueError("Paired inference requires ten training seeds")
    low, high = bootstrap_interval(delta)
    by_seed = delta.mean(axis=1)
    return dict(comparison=name, primary=primary, difference_pp=100*float(delta.mean()),
                ci_low_pp=100*low, ci_high_pp=100*high,
                paired_seed_p=sign_flip_pvalue(by_seed), primary_holm_3_p=None,
                positive_seeds=int(np.sum(by_seed > 0)), zero_seeds=int(np.sum(by_seed == 0)),
                negative_seeds=int(np.sum(by_seed < 0)), supports_primary_positive_difference=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    protocol_path = OUT / "confirmation_protocol.json"
    eval_path = OUT / "evaluation_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    evaluation_plan = json.loads(eval_path.read_text())
    if evaluation_plan["confirmation_protocol_sha256"] != digest(protocol_path):
        raise ValueError("Changed confirmation protocol")
    jobs = evaluation_plan["jobs"]
    paired_count = sum(bool(j.get("paired_replay")) for j in jobs)
    if (paired_count != protocol["paired_mechanism_jobs"] or
            len(jobs)-paired_count != protocol["ordinary_eval_count"] or
            len({j["output"] for j in jobs}) != len(jobs)):
        raise ValueError("Invalid matrix coverage or duplicate jobs")
    report = OUT / "report"
    report.mkdir(exist_ok=True)
    episodes, per_seed, missing = [], [], []
    grouped = {}
    for job in jobs:
        directory = ROOT / job["output"]
        if not (directory / "summary.json").exists():
            missing.append(job["output"])
            continue
        name = job["output"].split("/S03/")[1].replace("/", "__")
        summary = json.loads((directory / "summary.json").read_text())
        if summary["status"] != "completed" or summary["job_sha256"] != digest(OUT / "jobs" / (name+".json")):
            raise ValueError("Invalid completion: " + name)
        modes = ("nominal_closed", "nominal_replay", "pushed_closed", "pushed_replay") if job.get("paired_replay") else ("evaluation",)
        for mode in modes:
            data = json.loads((directory / mode / "episodes.json").read_text())
            if [d["seed"] for d in data] != list(range(job["eval_start"], job["eval_start"]+job["eval_count"])):
                raise ValueError("Test case identities/order differ")
            if any(not isinstance(d["clean_success30"], bool) for d in data):
                raise ValueError("Missing audited score")
            key = (job["condition"], job["group"], job["budget"], mode)
            grouped.setdefault(key, []).append((job["seed"], data))
            identifiers = dict(condition=key[0], group=key[1], budget=key[2], mode=mode,
                               train_seed=None if job["method"] == "zero" else job["seed"])
            for item in data:
                episodes.append(dict(**identifiers, **item))
            per_seed.append(dict(**identifiers, episodes=len(data),
                                 clean_success30=float(np.mean([d["clean_success30"] for d in data])),
                                 raw_success30=float(np.mean([d["success30"] for d in data])),
                                 full_episode_clean_success30=float(np.mean([d["full_episode_clean_success30"] for d in data])),
                                 mean_hold_sec=float(np.mean([d["longest_hold_sec"] for d in data])),
                                 mean_substep_peak_force_n=float(np.mean([d["substep_force_peak_n"] for d in data]))))
    if missing and not args.allow_partial:
        raise RuntimeError(f"Incomplete final matrix: {len(missing)} jobs")
    aggregate, matrices = [], {}
    for key, runs in sorted(grouped.items()):
        runs.sort(key=lambda x: x[0])
        if [r[0] for r in runs] != ([21001] if key[1] == "zero" else SEEDS):
            continue
        data = [d for _, rows in runs for d in rows]
        matrix = np.array([[d["clean_success30"] for d in rows] for _, rows in runs], float)
        matrices[key] = matrix
        low, high = bootstrap_interval(matrix)
        pre = [d for d in data if d["valid_before_push"]]
        raw_fail = sum(not d["success30"] for d in data)
        dirty = sum(d["success30"] and not d["clean_success30"] for d in data)
        aggregate.append(dict(condition=key[0], group=key[1], budget=key[2], mode=key[3],
                              policies=len(runs), episodes=len(data), clean_success30=float(matrix.mean()),
                              ci_low=low, ci_high=high, raw_success30=float(np.mean([d["success30"] for d in data])),
                              full_episode_clean_success30=float(np.mean([d["full_episode_clean_success30"] for d in data])),
                              mean_hold_sec=float(np.mean([d["longest_hold_sec"] for d in data])),
                              mean_substep_peak_force_n=float(np.mean([d["substep_force_peak_n"] for d in data])),
                              p95_substep_peak_force_n=float(np.quantile([d["substep_force_peak_n"] for d in data], .95)),
                              max_penetration_mm=1000*max(d["max_penetration_m"] for d in data),
                              raw_failures=raw_fail, raw_success_but_dirty=dirty,
                              clean_successes=int(matrix.sum()), pre_push_valid=len(pre),
                              post_push_valid_given_pre=float(np.mean([d["valid_one_second_after_push"] for d in pre])) if pre else None))
    comparisons = []
    for condition in protocol["benchmark_names"] + ["X_cuboid", "X_sphere"]:
        for budget in protocol["budgets"]:
            a = matrices.get((condition, "residual_tuned", budget, "evaluation"))
            for group in ("ppo_fixed", "ppo_tuned", "guided_tuned", "bc", "zero"):
                b = matrices.get((condition, group, 0 if group in ("bc", "zero") else budget, "evaluation"))
                if a is not None and b is not None:
                    primary = condition == "B0_nominal" and budget == 655360 and group in ("ppo_tuned", "guided_tuned")
                    comparisons.append(contrast(a, b, f"{condition}/{budget}/residual_tuned-{group}", primary))
    a = matrices.get(("mechanism", "residual_tuned", 655360, "pushed_closed"))
    b = matrices.get(("mechanism", "residual_tuned", 655360, "pushed_replay"))
    if a is not None and b is not None:
        comparisons.append(contrast(a, b, "mechanism/655360/pushed_closed-pushed_replay", True))
    for budget in protocol["budgets"]:
        a = matrices.get(("B0_nominal", "ppo_tuned", budget, "evaluation"))
        b = matrices.get(("B0_nominal", "ppo_fixed", budget, "evaluation"))
        if a is not None and b is not None:
            comparisons.append(contrast(a, b, f"B0_nominal/{budget}/ppo_tuned-ppo_fixed"))
    for group in ("ppo_fixed", "ppo_tuned", "guided_tuned", "residual_tuned"):
        a = matrices.get(("B0_nominal", group, 655360, "evaluation"))
        b = matrices.get(("B0_nominal", group, 163840, "evaluation"))
        if a is not None and b is not None:
            comparisons.append(contrast(a, b, f"B0_nominal/{group}/long-short"))
    primary = [r for r in comparisons if r["primary"]]
    if len(primary) == 3:
        for row, p in zip(primary, holm([r["paired_seed_p"] for r in primary])):
            row["primary_holm_3_p"] = p
            row["supports_primary_positive_difference"] = bool(p < .05 and row["ci_low_pp"] > 0)
    elif not args.allow_partial:
        raise ValueError("Missing planned primary contrasts")
    costs, curves, efficiency = [], [], []
    train_configs = list((OUT / "jobs").glob("development__*.json"))
    train_configs += list((OUT / "jobs").glob("confirmation__*.json"))
    for config in sorted(train_configs):
        job = json.loads(config.read_text())
        directory = ROOT / job["output"]
        if not (directory / "summary.json").exists():
            missing.append(job["output"])
            continue
        summary = json.loads((directory / "summary.json").read_text())
        if summary["job_sha256"] != digest(config):
            raise ValueError("Changed training configuration")
        costs.append(dict(stage=job["stage"], group=job.get("group", job["method"]), seed=job["seed"],
                          round=job.get("round"), online_steps=summary["final_step"],
                          elapsed_worker_sec=summary["elapsed_sec"], bc_elapsed_sec=summary["bc_elapsed_sec"],
                          first=summary["first"], last=summary["last"], max=summary["max"],
                          best_step=summary["best_step"], clean_auc=summary["clean_auc"],
                          job=job["output"]))
        with (directory / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            curves.append(dict(stage=job["stage"], group=job.get("group", job["method"]),
                               seed=job["seed"], round=job.get("round"), step=int(row["step"]),
                               clean_success30=float(row["clean_success30"]),
                               raw_success30=float(row["success30"]), mean_hold_sec=float(row["mean_hold"]),
                               elapsed_sec=float(row["elapsed_sec"])))
        if job["stage"] == "confirmation":
            for threshold in (.25, .5, .75):
                hits = [r for r in rows if float(r["clean_success30"]) >= threshold]
                efficiency.append(dict(group=job["group"], seed=job["seed"], threshold=threshold,
                                       reached=bool(hits), first_checkpoint_steps=int(hits[0]["step"]) if hits else None,
                                       censored_at_steps=job["steps"], development_cases=16))
    if missing and not args.allow_partial:
        raise RuntimeError("Some training or evaluation jobs are missing")
    for name, rows in (("episodes", episodes), ("per_seed", per_seed), ("aggregate", aggregate),
                       ("paired", comparisons), ("costs", costs), ("learning_curves", curves),
                       ("sample_efficiency", efficiency)):
        write_csv(report / (name+".csv"), rows)
    result = dict(complete=not missing, missing=missing, episodes=len(episodes), aggregate=aggregate,
                  comparisons=comparisons, confirmation_protocol_sha256=digest(protocol_path),
                  evaluation_protocol_sha256=digest(eval_path), source_sha256=digest(Path(__file__)),
                  preparation_cost="Existing video/reference preparation cost not included; unrecoverable human/LLM cost is unknown, not zero",
                  cost_units="elapsed_worker_sec includes BC; concurrent worker time is not wall time or GPU compute time",
                  inference="Holm correction only for three preregistered primary comparisons; others descriptive",
                  unit="ten training seeds crossed with 200 shared test initial states; one untrained reference controller")
    (report / "statistics.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["# S03 强基线、残差反馈与物体迁移", "",
             f"状态：{'完整矩阵统计完成' if not missing else '部分结果，不能作最终结论'}；统计{len(episodes)}个正式测试回合。", "",
             "主指标：撤托后连续有效保持至少30秒、末状态有效、全部5ms物理子步无禁接触且穿透不超过3mm。",
             "短/长预算来自同一次训练的预定检查点，不是独立实验。所有方法在方块上训练，新形状不再训练。", "",
             "## 汇总", "", "| 条件 | 方法 | 在线步数 | 模式 | 清洁成功率[95%区间] | 原始成功率 | 全程穿透合格成功率 |",
             "|---|---|---|---|---|---|---|"]
    for r in aggregate:
        lines.append(f"| {r['condition']} | {NAMES[r['group']]} | {r['budget']} | {r['mode']} | "
                     f"{100*r['clean_success30']:.2f}% [{100*r['ci_low']:.2f}, {100*r['ci_high']:.2f}] | "
                     f"{100*r['raw_success30']:.2f}% | {100*r['full_episode_clean_success30']:.2f}% |")
    lines += ["", "## 三项主要比较", "", "| 比较 | 差值百分点[95%区间] | Holm-3 p | 支持正向收益 |",
              "|---|---|---|---|"]
    for r in primary:
        lines.append(f"| {r['comparison']} | {r['difference_pp']:+.2f} [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}] | "
                     f"{r['primary_holm_3_p']} | {r['supports_primary_positive_difference']} |")
    lines += ["", "## 复算资料与边界", "",
              "- episodes.csv保留逐回合得分、保持时间、力、穿透和扰动前后有效性；aggregate.csv包含失败分解。",
              "- paired.csv完整列出所有扩展对照；仅三项预定比较进行主要假设检验，不能筛选描述性p值宣布获胜。",
              "- costs.csv保留全部开发候选，包括失败候选；BC耗时已包含在总进程耗时内，不能重复相加。",
              "- sample_efficiency.csv是16个开发初态上的阈值到达检查点；未达到的模型保留为右删失，不删除。",
              "- 固定回放来自无扰动轨迹；源回合终止后修正置零，失败后尾段不计入有效控制证据。",
              "- 主要清洁指标只约束撤托后；全程穿透合格成功率另作诊断，不替换评分规则。",
              "- 一条视频的校正参考不是实时视觉网络；跨物体迁移也不等于跨视频泛化。",
              "- 视频/参考准备和历史调试成本不能忽略；无法核实的人力/大模型费用标记未知。",
              "- 新物体没有形状条件输入，也没有重新重定向；失败同样是本实验结果。",
              "- 不支持优越性不代表证明等效；模拟接触合格不代表真机安全。"]
    (report / "results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(complete=not missing, episodes=len(episodes), missing=len(missing))))


if __name__ == "__main__":
    main()
