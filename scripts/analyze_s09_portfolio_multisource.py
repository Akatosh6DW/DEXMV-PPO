"""Audit and summarize the precommitted three-lineage historical portfolios."""

import json
from pathlib import Path

import numpy as np

from analyze_s09_bc_confirmation import bootstrap, plt, write_csv
from analyze_s09_late_experts import transitions
from analyze_s09_ppo_confirmation import failure_gate_summary
from confirm_s09_portfolio_multisource import COUNT, GROUPS, OUT, ROOT, SOURCES, START, read
from make_s09_checkpoint_portfolio import choose_sources
from retarget_s09_reference import digest

LABELS = dict(bc="Original BC", stage="Stage BC+PPO", expert_last="Expert endpoint", portfolio="Historical portfolio")


def main():
    plan = read(OUT / "plan.json")
    assert (plan["eval_start"], plan["eval_count"]) == (START, COUNT)
    assert len(plan["jobs"]) == 12
    assert {j["name"] for j in plan["jobs"]} == {f"{g}_s{s}" for s, _ in SOURCES for g in GROUPS}
    for path, sha in plan["execution_sources"].items():
        assert digest(ROOT / path) == sha, path
    selections = []
    for source, _ in SOURCES:
        binding = plan["selections"][str(source)]
        assert digest(ROOT / binding["path"]) == binding["sha256"]
        selection = read(ROOT / binding["path"])
        assert selection["development_seeds"] == [130000, 130063]
        assert selection["extra_ppo_steps"] == 0
        assert selection["selected_source_by_mode"] == [GROUPS[k] for k in choose_sources(selection["source_success_counts"])]
        for item in selection["sources"]:
            path = ROOT / item["checkpoint"]
            assert digest(path) == item["checkpoint_sha256"]
            assert digest(path.parent / "episodes.json") == item["episodes_sha256"]
        selections.append(selection)
    raw, models, modes, gates, evidence, groups = [], [], [], [], {}, {}
    initial = None
    for job in plan["jobs"]:
        run = ROOT / job["output"]
        cfg, summary = read(run / "run_config.json"), read(run / "summary.json")
        assert summary["status"] == "completed" and summary["final_step"] == cfg["steps"] == 0
        assert cfg["method"] == "guided" and cfg["seed"] == job["training_seed"]
        for key in ("eval_start", "eval_count", "model_sha256", "reference_sha256", "env_config"):
            assert cfg[key] == plan[key], key
        assert cfg["checkpoint_sha256"] == job["checkpoint_sha256"] == digest(ROOT / job["checkpoint"])
        assert digest(ROOT / job["source_config"]) == job["source_config_sha256"]
        assert read(run / "quality_config.json")["primary_success_unchanged"]
        assert read(run / "failure_config.json")["original_success_unchanged"]
        folder = run / "step_000000000"
        audit = read(folder / "native_replay_audit.json")
        assert audit["status"] == "verified" and audit["exact_original_evaluation_reproduced"]
        assert audit["normalized_action_tolerance"] == 1e-5
        assert audit["checkpoint_sha256"] == digest(folder / "params.pkl")
        assert audit["trace_sha256"] == digest(folder / "preview_trajectories.npz")
        with np.load(folder / "initial_states.npz") as z:
            np.testing.assert_array_equal(z["seeds"], np.arange(START, START+COUNT))
            if initial is None:
                initial = z["qpos"].copy()
            np.testing.assert_array_equal(z["qpos"], initial)
        data = read(folder / "episodes.json")
        assert [r["seed"] for r in data] == list(range(START, START+COUNT))
        for r in data:
            assert r["released_rope_force_n"] == r["residual_rms_rad"] == 0
            if r["success30"]:
                assert r["hold_sec"] >= 30 and r["final_valid"] and not r["forbidden"]
                assert r["initial_penetration_m"] <= .001 and r["max_released_penetration_m"] <= .003
        assert np.mean([r["success30"] for r in data]) == summary["rows"][0]["success30"]
        key = (job["training_source"], job["group"])
        assert key not in groups
        groups[key] = data
        models.append(dict(training_source=key[0], group=key[1], successes=sum(r["success30"] for r in data),
            n=COUNT, success_rate=float(np.mean([r["success30"] for r in data])),
            mean_hold_sec=float(np.mean([r["hold_sec"] for r in data])),
            max_released_depth_mm=1000*max(r["max_released_penetration_m"] for r in data),
            max_all_depth_mm=1000*max(r["max_all_penetration_m"] for r in data),
            peak_force_n=max(r["peak_force_n"] for r in data),
            successful_peak_force_n=max((r["peak_force_n"] for r in data if r["success30"]), default=0),
            all_depth_successes=sum(r["success30_all_depth_le_3mm"] for r in data),
            path_online_steps=job["path_online_steps"], actor_parameter_multiplier=job["actor_parameter_multiplier"]))
        raw.extend(dict(training_source=key[0], group=key[1], **r) for r in data)
        gates.extend(dict(training_source=key[0], group=key[1], **r) for r in failure_gate_summary(data))
        for mode in range(6):
            part = [r for r in data if r["selected_reference"] == mode]
            modes.append(dict(training_source=key[0], group=key[1], mode=mode,
                n=len(part), successes=sum(r["success30"] for r in part)))
        evidence[job["name"]] = {name: digest(folder / name) for name in
            ("episodes.json", "initial_states.npz", "params.pkl", "native_replay_audit.json")}
    matrices = {g: np.asarray([[r["success30"] for r in groups[(s, g)]] for s, _ in SOURCES], float) for g in GROUPS}
    summaries, contrasts, paired = [], [], []
    for group, matrix in matrices.items():
        summaries.append(dict(group=group, success_rate=float(matrix.mean()), successes=int(matrix.sum()),
            policy_state_pairs=matrix.size, unique_states=COUNT, training_sources=len(SOURCES),
            per_source=matrix.mean(axis=1).tolist(), descriptive_ci95=bootstrap(matrix, draws=20000)))
    for group in GROUPS[:-1]:
        x, y = matrices["portfolio"], matrices[group]
        contrasts.append(dict(contrast=f"portfolio minus {group}", difference=float((x-y).mean()),
            per_source_difference=(x-y).mean(axis=1).tolist(),
            descriptive_ci95=bootstrap(x, y, paired_models=True, draws=20000)))
        for source, _ in SOURCES:
            paired.append(dict(training_source=source, baseline=group,
                **transitions(groups[(source, group)], groups[(source, "portfolio")])))
    out = OUT / "report"
    out.mkdir(exist_ok=True)
    for name, rows in (("models.csv", models), ("episodes.csv", raw), ("orientation.csv", modes),
                       ("failure_gates.csv", gates), ("paired_successes.csv", paired)):
        write_csv(out / name, rows)
    stats = dict(status="completed_precommitted_confirmation", plan_sha256=digest(OUT / "plan.json"),
        analysis_source_sha256=digest(Path(__file__)), summaries=summaries, contrasts=contrasts,
        paired=paired, models=models, selections=selections, evidence=evidence,
        initial_states_exactly_paired=True, additional_ppo_steps=0,
        uncertainty="Descriptive paired lineage/common-state bootstrap; three lineages, not 768 independent trainings",
        selection="Fixed historical selection on old development cases only; no new test-driven reselection")
    (out / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for i, (row, color) in enumerate(zip(summaries, ("#777777", "#c89535", "#b65b50", "#27816e"))):
        ax.bar(i, 100*row["success_rate"], color=color, alpha=.75)
        ax.plot([i, i], 100*np.asarray(row["descriptive_ci95"]), color="black", linewidth=1)
        ax.scatter(np.linspace(i-.12, i+.12, 3), 100*np.asarray(row["per_source"]), color="black", s=20)
    ax.set(xticks=range(4), xticklabels=[LABELS[g] for g in GROUPS], ylim=(0, 100),
        ylabel="Strict 30s success (%)", title="Fixed historical selection: 3 lineages, 256 shared new states")
    fig.savefig(out / "heldout_success.png", dpi=160)
    plt.close(fig)
    lines = ["# 历史组合：三个来源的新初态确认", "",
        "只用旧开发集固定选取各朝向的BC/阶段/专家末步actor，再评估200000--200255。未按本测试重选。",
        "每组768策略-初态对，只有三个训练来源和256个唯一初态；组合新增0步PPO，继承1310720步，actor六倍容量。", "",
        "| 方法 | 平均成功率 | 三来源分别 | 成功/执行 | 描述性95%区间 |", "|---|---:|---|---:|---|"]
    for r in summaries:
        lo, hi = r["descriptive_ci95"]
        lines.append(f"| {LABELS[r['group']]} | {r['success_rate']:.2%} | " +
            " / ".join(f"{v:.2%}" for v in r["per_source"]) + f" | {r['successes']}/768 | [{lo:.2%}, {hi:.2%}] |")
    lines += ["", "![确认结果](heldout_success.png)", "", "## 配对差值", ""]
    for r in contrasts:
        lo, hi = r["descriptive_ci95"]
        lines.append(f"- {r['contrast']}：{100*r['difference']:+.2f}个百分点，描述性95%区间[{100*lo:+.2f}, {100*hi:+.2f}]。")
    lines += ["", "## 限制", "",
        "- 这是固定历史组合规则的重复验证，不是新训练算法，也不是执行级残差。不能用它覆盖专家末步确认的负结果。",
        "- 只有三个来源；区间不是所有初始化有效的保证，额外开发选择和六倍actor成本必须计入。",
        "- 阶段起点也是BC+PPO，累计655360步；专家和组合继承1310720步。只有专家与组合路径预算相同。",
        "- 成功仍按释放后3mm等原严格判据；全回合3mm、失败载荷和穿透另见models.csv，不声称硬件安全。",
        "- 所有保存预览均原批次重现并独立物理核验，不声称全部未保存回合都另做回放。",
        "- 此测试区间已消耗，不能用于下一次改法的独立确认。",
        "", "逐例episodes.csv，分朝向orientation.csv，配对得失paired_successes.csv，接触models.csv，证据statistics.json。"]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(summaries=summaries, contrasts=contrasts)), flush=True)


if __name__ == "__main__":
    main()
