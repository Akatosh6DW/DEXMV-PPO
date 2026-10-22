"""Analyze all precommitted test endpoints with lineage/state paired intervals."""

import json
from pathlib import Path

import numpy as np

from analyze_s09_bc_confirmation import bootstrap, plt, write_csv
from analyze_s09_late_experts import transitions
from analyze_s09_ppo_confirmation import failure_gate_summary
from confirm_s09_experts_multisource import COUNT, OUT, ROOT, SOURCES, START
from retarget_s09_reference import digest

GROUPS = ("bc", "stage", "single_last", "expert_last")
LABELS = {"bc": "Original BC", "stage": "Stage start", "single_last": "Single BC+PPO", "expert_last": "Expert BC+PPO"}


def main():
    read = lambda p: json.loads(p.read_text())
    plan = read(OUT / "plan.json")
    assert (plan["eval_start"], plan["eval_count"]) == (START, COUNT)
    assert len(plan["jobs"]) == 12
    expected_jobs = {f"{g}_s{s}" for s, _ in SOURCES for g in GROUPS}
    assert {j["name"] for j in plan["jobs"]} == expected_jobs
    for path, sha in plan["execution_sources"].items():
        assert digest(ROOT / path) == sha, path
    models, raw, modes, gates, groups, evidence = [], [], [], [], {}, {}
    initial = None
    for job in plan["jobs"]:
        run = ROOT / job["output"]
        cfg, summary = read(run / "run_config.json"), read(run / "summary.json")
        assert summary["status"] == "completed" and summary["final_step"] == cfg["steps"] == 0
        for key in ("eval_start", "eval_count", "model_sha256", "reference_sha256", "env_config"):
            assert cfg[key] == plan[key], key
        binding = read(OUT / "execution" / f"{job['name']}_binding.json")
        assert binding["plan_sha256"] == digest(OUT / "plan.json")
        assert binding["checkpoint"] == job["checkpoint"]
        assert binding["checkpoint_sha256"] == digest(ROOT / job["checkpoint"]) == cfg["checkpoint_sha256"]
        assert binding["source_run_config_sha256"] == job["source_run_config_sha256"]
        assert digest((ROOT / job["checkpoint"]).parent.parent / "run_config.json") == job["source_run_config_sha256"]
        assert read(run / "quality_config.json")["primary_success_unchanged"]
        assert read(run / "failure_config.json")["original_success_unchanged"]
        folder = run / "step_000000000"
        audit = read(folder / "native_replay_audit.json")
        assert audit["status"] == "verified" and audit["normalized_action_tolerance"] == 1e-5
        assert audit["exact_original_evaluation_reproduced"]
        assert audit["checkpoint_sha256"] == digest(folder / "params.pkl")
        assert audit["trace_sha256"] == digest(folder / "preview_trajectories.npz")
        with np.load(folder / "initial_states.npz") as states:
            np.testing.assert_array_equal(states["seeds"], np.arange(START, START + COUNT))
            if initial is None:
                initial = states["qpos"].copy()
            np.testing.assert_array_equal(states["qpos"], initial)
        episodes = read(folder / "episodes.json")
        assert [r["seed"] for r in episodes] == list(range(START, START + COUNT))
        for r in episodes:
            assert r["released_rope_force_n"] == r["residual_rms_rad"] == 0
            if r["success30"]:
                assert r["hold_sec"] >= 30 and r["final_valid"] and not r["forbidden"]
                assert r["initial_penetration_m"] <= .001 and r["max_released_penetration_m"] <= .003
        assert np.mean([r["success30"] for r in episodes]) == summary["rows"][0]["success30"]
        key = (job["training_source"], job["group"])
        assert key not in groups
        groups[key] = episodes
        models.append(dict(training_source=key[0], group=key[1], successes=sum(r["success30"] for r in episodes),
            n=COUNT, success_rate=float(np.mean([r["success30"] for r in episodes])),
            mean_hold_sec=float(np.mean([r["hold_sec"] for r in episodes])),
            max_released_depth_mm=1000 * max(r["max_released_penetration_m"] for r in episodes),
            max_all_depth_mm=1000 * max(r["max_all_penetration_m"] for r in episodes),
            peak_force_n=max(r["peak_force_n"] for r in episodes),
            successful_peak_force_n=max((r["peak_force_n"] for r in episodes if r["success30"]), default=0),
            all_depth_success_rate=float(np.mean([r["success30_all_depth_le_3mm"] for r in episodes])),
            path_online_steps=job["path_online_steps"], actor_parameter_multiplier=job["actor_parameter_multiplier"]))
        raw.extend(dict(training_source=key[0], group=key[1], **r) for r in episodes)
        gates.extend(dict(training_source=key[0], group=key[1], **r) for r in failure_gate_summary(episodes))
        for mode in range(6):
            part = [r for r in episodes if r["selected_reference"] == mode]
            modes.append(dict(training_source=key[0], group=key[1], mode=mode, n=len(part),
                              successes=sum(r["success30"] for r in part)))
        evidence[job["name"]] = dict(config_sha256=digest(run / "run_config.json"),
            episodes_sha256=digest(folder / "episodes.json"), audit_sha256=digest(folder / "native_replay_audit.json"),
            binding_sha256=digest(OUT / "execution" / f"{job['name']}_binding.json"))
    matrices = {g: np.asarray([[r["success30"] for r in groups[(s, g)]] for s, _ in SOURCES], float) for g in GROUPS}
    summaries, contrasts, paired = [], [], []
    for group, matrix in matrices.items():
        summaries.append(dict(group=group, success_rate=float(matrix.mean()), successes=int(matrix.sum()),
            policy_state_pairs=matrix.size, unique_states=COUNT, training_sources=3,
            per_source=matrix.mean(axis=1).tolist(), descriptive_ci95=bootstrap(matrix, draws=20000)))
    for group in ("bc", "stage", "single_last"):
        x, y = matrices["expert_last"], matrices[group]
        contrasts.append(dict(contrast=f"expert_last minus {group}", difference=float((x-y).mean()),
            per_source_difference=(x-y).mean(axis=1).tolist(),
            descriptive_ci95=bootstrap(x, y, paired_models=True, draws=20000)))
        for source, _ in SOURCES:
            paired.append(dict(training_source=source, baseline=group,
                **transitions(groups[(source, group)], groups[(source, "expert_last")])))
    out = OUT / "report"
    out.mkdir(exist_ok=True)
    for name, data in (("models.csv", models), ("episodes.csv", raw), ("orientation.csv", modes),
                       ("failure_gates.csv", gates), ("paired_successes.csv", paired)):
        write_csv(out / name, data)
    stats = dict(status="completed_precommitted_confirmation", plan_sha256=digest(OUT / "plan.json"),
        source_sha256=digest(Path(__file__)), summaries=summaries, contrasts=contrasts, paired=paired,
        models=models, evidence=evidence, initial_states_exactly_paired=True,
        uncertainty="Descriptive paired lineage and common-state bootstrap; only three lineages, not 768 independent trainings",
        selection="All precommitted true endpoints; no test-driven checkpoint selection")
    (out / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    colors = ("#777777", "#c89535", "#b65b50", "#27816e")
    for i, row in enumerate(summaries):
        ax.bar(i, 100 * row["success_rate"], color=colors[i], alpha=.75)
        lo, hi = np.asarray(row["descriptive_ci95"]) * 100
        ax.plot([i, i], [lo, hi], color="black", linewidth=1)
        ax.scatter(np.linspace(i-.12, i+.12, 3), 100 * np.asarray(row["per_source"]), color="black", s=20)
    ax.set(xticks=range(4), xticklabels=[LABELS[g] for g in GROUPS], ylim=(0, 100),
           ylabel="Strict 30s success (%)", title="3 training lineages, 256 shared new test states")
    fig.savefig(out / "heldout_success.png", dpi=160)
    plt.close(fig)
    lines = ["# 专家BC+PPO：三个来源的新初态确认", "",
        "三个来源、同256个未用于本轮调参的新初态，四组全部按预定路径报告。每组768策略-初态对，不是768个训练种子。",
        "专家和单网络均报告真正末步，原BC为0在线步，阶段起点655360步，两个末步均累计1310720步。", "",
        "| 方法 | 总成功率 | 三来源分别 | 成功/策略-初态对 | 描述性95%区间 |",
        "|---|---:|---|---:|---|"]
    for r in summaries:
        lo, hi = r["descriptive_ci95"]
        lines.append(f"| {LABELS[r['group']]} | {r['success_rate']:.2%} | " +
            " / ".join(f"{v:.2%}" for v in r["per_source"]) + f" | {r['successes']}/768 | [{lo:.2%}, {hi:.2%}] |")
    lines += ["", "![新初态确认](heldout_success.png)", "", "## 配对差值", ""]
    for r in contrasts:
        lo, hi = r["descriptive_ci95"]
        lines.append(f"- {r['contrast']}：{100*r['difference']:+.2f}个百分点，描述性95%区间[{100*lo:+.2f}, {100*hi:+.2f}]。")
    lines += ["", "## 证据边界", "",
        "- 三个来源的BC、阶段起点、单网络与专家按祖先配对。图中黑点是三个来源，不是回合重复。",
        "- 来源与初态共同重采样，来源只有三个，区间不能证明所有训练种子都有效。",
        "- 专家actor为六倍参数量；同在线步数不是同计算量或同调参总成本。",
        "- 未更换视频、物理或成功判据；当前实验不回答残差是否优于其参考基础。",
        "- 主成功要求释放后穿透不超过3mm；全回合3mm附加结果、失败原因及峰值力另列models.csv和failure_gates.csv。",
        "- 保存预览均原批次重现并独立物理核验，不声称全部未保存执行都另做回放，更不等于实机安全验证。",
        "- 此区间从现在起已消耗，不再用于下一轮调参后的独立确认。",
        "", "原始逐例episodes.csv，来源汇总models.csv，配对得失paired_successes.csv，完整统计及哈希statistics.json。"]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(summaries=summaries, contrasts=contrasts)), flush=True)


if __name__ == "__main__":
    main()
