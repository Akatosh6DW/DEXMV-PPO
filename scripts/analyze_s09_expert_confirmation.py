"""Paired heldout analysis for a frozen, single-origin expert endpoint."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from analyze_s09_bc_confirmation import bootstrap, write_csv, plt
from analyze_s09_ppo_confirmation import transitions, failure_gate_summary
from confirm_s09_late_experts import OUT, ROOT
from retarget_s09_reference import digest


def holm(pvalues):
    pvalues = np.asarray(pvalues, float)
    if pvalues.ndim != 1 or not len(pvalues) or not np.isfinite(pvalues).all() or np.any((pvalues < 0) | (pvalues > 1)):
        raise ValueError("Require finite probability values")
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1., np.maximum.accumulate(pvalues[order]*np.arange(len(order), 0, -1)))
    return adjusted.tolist()


def main():
    read = lambda p: json.loads(p.read_text())
    plan = read(OUT/"plan.json")
    for path, value in plan["execution_sources"].items():
        assert digest(ROOT/path) == value, path
    expected = list(range(plan["eval_start"], plan["eval_start"]+plan["eval_count"]))
    summaries, episodes, modes, gates, matrices, evidence = [], [], [], [], {}, {}
    initial = None
    for job in plan["jobs"]:
        run = ROOT/job["output"]
        cfg, summary = read(run/"run_config.json"), read(run/"summary.json")
        assert summary["status"] == "completed" and summary["final_step"] == cfg["steps"] == 0
        for key in ("eval_start", "eval_count", "model_sha256", "reference_sha256", "env_config"):
            assert cfg[key] == plan[key], key
        assert cfg["checkpoint_sha256"] == job["checkpoint_sha256"] == digest(ROOT/job["checkpoint"])
        assert read(run/"quality_config.json")["primary_success_unchanged"]
        assert read(run/"failure_config.json")["original_success_unchanged"]
        folder = run/"step_000000000"
        audit = read(folder/"native_replay_audit.json")
        assert audit["status"] == "verified" and audit["normalized_action_tolerance"] == 1e-5
        assert audit["checkpoint_sha256"] == digest(folder/"params.pkl")
        assert audit["trace_sha256"] == digest(folder/"preview_trajectories.npz")
        with np.load(folder/"initial_states.npz") as z:
            np.testing.assert_array_equal(z["seeds"], expected)
            if initial is None:
                initial = z["qpos"].copy()
            np.testing.assert_array_equal(z["qpos"], initial)
        rows = read(folder/"episodes.json")
        assert [r["seed"] for r in rows] == expected
        for row in rows:
            assert row["released_rope_force_n"] == row["residual_rms_rad"] == 0
            if row["success30"]:
                assert row["hold_sec"] >= 30 and row["final_valid"] and not row["forbidden"]
                assert row["max_released_penetration_m"] <= .003 and row["initial_penetration_m"] <= .001
        array = np.asarray([[r["success30"] for r in rows]], float)
        assert array.mean() == summary["rows"][0]["success30"]
        matrices[job["name"]] = array
        ci = binomtest(int(array.sum()), array.size).proportion_ci(method="wilson")
        summaries.append(dict(group=job["name"], successes=int(array.sum()), n=array.size,
            success_rate=float(array.mean()), ci95=[float(ci.low), float(ci.high)],
            mean_hold_sec=float(np.mean([r["hold_sec"] for r in rows])),
            max_released_depth_mm=1000*max(r["max_released_penetration_m"] for r in rows),
            max_all_depth_mm=1000*max(r["max_all_penetration_m"] for r in rows),
            successful_peak_force_n=max((r["peak_force_n"] for r in rows if r["success30"]), default=None),
            peak_force_n=max(r["peak_force_n"] for r in rows),
            all_depth_success_rate=float(np.mean([r["success30_all_depth_le_3mm"] for r in rows])),
            path_online_steps=job["path_online_steps"], actor_parameter_multiplier=job["actor_parameter_multiplier"]))
        for mode in range(6):
            subset = [r for r in rows if r["selected_reference"] == mode]
            modes.append(dict(group=job["name"], mode=mode, n=len(subset),
                successes=sum(r["success30"] for r in subset)))
        gates.extend(dict(group=job["name"], **r) for r in failure_gate_summary(rows))
        episodes.extend(dict(group=job["name"], **r) for r in rows)
        evidence[job["name"]] = dict(config_sha256=digest(run/"run_config.json"),
            episodes_sha256=digest(folder/"episodes.json"), audit_sha256=digest(folder/"native_replay_audit.json"))
    differences = []
    for name in ("bc", "stage", "single_last"):
        x, y = matrices["expert_last"], matrices[name]
        pairing = transitions(y, x)
        discordant = pairing["gained"]+pairing["lost"]
        pvalue = binomtest(pairing["gained"], discordant).pvalue if discordant else 1.
        differences.append(dict(contrast="expert_last minus "+name,
            difference=float((x-y).mean()), ci95=bootstrap(x, y, paired_models=True, draws=20000),
            exact_mcnemar_p=float(pvalue), **pairing))
    for row, p in zip(differences, holm([r["exact_mcnemar_p"] for r in differences])):
        row["holm_p_three_comparisons"] = p
    output = OUT/"report"
    output.mkdir(exist_ok=True)
    result = dict(status="completed_one_origin_confirmation", plan_sha256=digest(OUT/"plan.json"),
        summaries=summaries, differences=differences, evidence=evidence, source_sha256=digest(Path(__file__)),
        limitations=plan["limitations"], interval_scope="Conditional on fixed policies; initial-state uncertainty, not training-seed variance")
    (output/"statistics.json").write_text(json.dumps(result, indent=2)+"\n")
    (output/"episodes.json").write_text(json.dumps(episodes, indent=2)+"\n")
    for name, rows in (("summary.csv", summaries), ("paired.csv", differences), ("orientation.csv", modes), ("failure_gates.csv", gates)):
        write_csv(output/name, rows)
    write_csv(output/"episodes.csv", [dict(r, failure_gates=json.dumps(r["failure_gates"], sort_keys=True)) for r in episodes])
    fig, ax = plt.subplots(figsize=(8,4.5), layout="constrained")
    for index, row in enumerate(summaries):
        ax.bar(index, row["success_rate"]*100, color=["#777777", "#c48a25", "#418b91", "#a94b55"][index])
        ax.plot([index,index], 100*np.asarray(row["ci95"]), color="black")
    ax.set(xticks=range(4), xticklabels=["BC", "Stage start", "Single last", "Experts last"],
        ylim=(0,100), ylabel="Strict success (%)", title="256 paired unseen initial states; one training origin")
    fig.savefig(output/"success.png", dpi=160)
    plt.close(fig)
    lines = ["# 分专家BC+PPO末步：256个新初态配对确认", "",
        "四组权重在新测试前冻结。只评估真实末步，不拿开发集最佳中间点替换。主任务仍是随机三维朝向圆柱、3秒撤支撑、严格连续30秒且末帧合格。", "",
        "| 组 | 成功 | 成功率 | 平均合格秒 | 全案例释放后最大穿透mm | 成功案例峰值N | 累计在线步 | actor倍数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['group']} | {r['successes']}/{r['n']} | {r['success_rate']:.2%} | {r['mean_hold_sec']:.3f} | {r['max_released_depth_mm']:.3f} | {r['successful_peak_force_n']} | {r['path_online_steps']} | {r['actor_parameter_multiplier']} |")
    lines += ["", "| 专家减基线 | 成功率差 | 配对95%区间 | 新增/丢失 | Holm校正p |", "|---|---:|---:|---:|---:|"]
    for r in differences:
        lines.append(f"| {r['contrast']} | {r['difference']:.2%} | [{r['ci95'][0]:.2%}, {r['ci95'][1]:.2%}] | {r['gained']}/{r['lost']} | {r['holm_p_three_comparisons']:.5g} |")
    lines += ["", "## 如何解读", "",
        "区间是对同一对固定模型按共同初态进行20000次配对bootstrap；McNemar双侧精确检验并对三个比较作Holm校正。Wilson区间见summary.csv。它们均不覆盖训练种子、调参过程或跨任务的不确定性。",
        "只有一个BC/训练来源，专家actor六倍容量。不能把256回合称为256个训练重复，不能据此声称算法普遍更优或完成所有稳定性验证。E79/E80短重复的不一致不会被本表覆盖。",
        "最大穿透和峰值统计保留失败案例，另给成功案例峰值。原3mm主门槛没有放宽；全回合3mm是额外诊断，不能替代主指标。质量与失效门槛对全部256例统计，原生回放覆盖保存预览，不冒充全例另行重放。",
        "这是分专家完整动作BC+PPO，不是视频参考加残差，也不是反馈BC纠错数据实验。原始逐例JSON/CSV、六朝向计数、失效门槛和文件哈希都在本目录。"]
    (output/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
