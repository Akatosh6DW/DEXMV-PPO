"""Combine three completed expert endpoints, preserving training-source pairing."""

import csv
import json
from pathlib import Path

import numpy as np

from analyze_s09_bc_confirmation import bootstrap, plt, write_csv
from analyze_s09_late_experts import BASE, select_runs, transitions
from retarget_s09_reference import digest


def main():
    read = lambda p: json.loads(p.read_text())
    protocols = [
        (14121, select_runs(long_replica=14121), "E19_bc_fast_critic_long", "I26_phase_s13101"),
        (14122, select_runs(long=True), "E27_bc_fast_critic_s14102", "I27_phase_s13102"),
        (14123, select_runs(long_replica=14123), "E29_bc_fast_critic_s14103", "I28_phase_s13103"),
    ]
    rows, modes, episodes, curves, evidence = [], [], [], [], {}
    matrices = {k: [] for k in ("bc", "stage", "single_last", "expert_last")}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained", sharey=True)
    for ax, (seed, (run, control, report, steps), bc_run, bc_origin) in zip(axes, protocols):
        assert steps == 655360
        individual = read(BASE / "report" / report / "statistics.json")
        assert individual["status"] == "completed_development_only"
        assert individual["run_config_sha256"] == digest(run / "run_config.json")
        cfg, bc_cfg = read(run / "run_config.json"), read(BASE / bc_run / "run_config.json")
        origin = BASE.parent / bc_origin / "step_000000000/params.pkl"
        assert bc_cfg["checkpoint_sha256"] == digest(origin)
        assert Path(bc_cfg["checkpoint"]).resolve() == origin.resolve()
        for key in ("model_sha256", "reference_sha256", "env_config", "eval_count", "eval_start"):
            assert cfg[key] == bc_cfg[key], key
        folders = {"bc": BASE / bc_run / "step_000000000", "stage": run / "step_000000000",
                   "single_last": control / "step_000655360", "expert_last": run / "step_000655360"}
        groups = {}
        for group, folder in folders.items():
            data = read(folder / "episodes.json")
            assert len(data) == 64 and [r["seed"] for r in data] == list(range(130000, 130064))
            with np.load(folder / "initial_states.npz") as current, np.load(folders["stage"] / "initial_states.npz") as initial:
                for key in ("seeds", "qpos"):
                    np.testing.assert_array_equal(current[key], initial[key])
            for r in data:
                assert r["released_rope_force_n"] == r["residual_rms_rad"] == 0
                if r["success30"]:
                    assert r["hold_sec"] >= 30 and r["final_valid"] and not r["forbidden"]
                    assert r["initial_penetration_m"] <= .001 and r["max_released_penetration_m"] <= .003
            groups[group] = data
            matrices[group].append([r["success30"] for r in data])
            episodes.extend(dict(training_source=seed, group=group, **r) for r in data)
            evidence[f"{seed}/{group}"] = {name: digest(folder / name) for name in (
                "params.pkl", "episodes.json", "initial_states.npz")}
        for mode in range(6):
            part = {g: [r for r in data if r["selected_reference"] == mode] for g, data in groups.items()}
            modes.append(dict(training_source=seed, mode=mode, n=len(part["bc"]),
                              **{g: sum(r["success30"] for r in data) for g, data in part.items()}))
        row = dict(training_source=seed, run=run.name,
            bc_successes=sum(r["success30"] for r in groups["bc"]),
            stage_successes=sum(r["success30"] for r in groups["stage"]),
            single_successes=sum(r["success30"] for r in groups["single_last"]),
            expert_successes=sum(r["success30"] for r in groups["expert_last"]),
            first=individual["first"], last=individual["last"], max=individual["max"], best_step=individual["best_step"])
        for name in ("bc", "stage", "single_last"):
            row.update({f"vs_{name}_{k}": v for k, v in transitions(groups[name], groups["expert_last"]).items()})
        row.update(max_depth_mm=1000 * max(r["max_released_penetration_m"] for r in groups["expert_last"]),
                   peak_force_n=max(r["peak_force_n"] for r in groups["expert_last"]))
        rows.append(row)
        with (BASE / "report" / report / "curve.csv").open() as stream:
            curve = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(stream)]
        curves.extend(dict(training_source=seed, **r) for r in curve)
        x = [r["step"] for r in curve]
        ax.plot(x, [100 * r["expert_success"] for r in curve], label="Expert BC+PPO", color="#27816e")
        ax.plot(x, [100 * r["control_success"] for r in curve], label="Single BC+PPO", color="#ba5a48")
        ax.axhline(100 * row["bc_successes"] / 64, label="Original BC reference", color="#666666", linestyle="--")
        ax.set(title=f"Training source {seed}", xlabel="Additional stage interactions", ylim=(0, 100))
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    axes[0].set_ylabel("Strict 30s success (%)")
    axes[-1].legend(fontsize=8)
    matrices = {k: np.asarray(v, float) for k, v in matrices.items()}
    contrasts = []
    for name in ("bc", "stage", "single_last"):
        a, b = matrices["expert_last"], matrices[name]
        contrasts.append(dict(contrast=f"expert_last minus {name}", difference=float((a-b).mean()),
            descriptive_ci95=bootstrap(a, b, paired_models=True, draws=20000),
            per_source_difference=(a-b).mean(axis=1).tolist()))
    out = BASE / "report/experts_multisource_long"
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("overview.csv", rows), ("orientation.csv", modes), ("episodes.csv", episodes), ("curves.csv", curves)):
        write_csv(out / name, data)
    fig.savefig(out / "development_curves.png", dpi=160)
    plt.close(fig)
    stats = dict(status="completed_development_only", training_sources=3, unique_development_states=64,
        policy_state_pairs_per_group=192, contrasts=contrasts, rows=rows, evidence=evidence,
        group_mean_rates={k: float(v.mean()) for k, v in matrices.items()},
        inherited_online_steps=655360, new_stage_online_steps=655360, total_path_online_steps=1310720,
        source_sha256=digest(Path(__file__)), limitations="Three sources, reused dev states, sixfold actor capacity; not heldout")
    (out / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    lines = ["# 专家BC+PPO：三个来源的相同长阶段", "",
        "全部按655360新增在线步的真正末步比较，继承阶段655360步，总路径1310720步。",
        "同64个反复使用的开发初态，3个训练来源；192策略-初态对不是192独立训练。", "",
        "| 来源 | 原BC | 阶段起点 | 同预算单网络末步 | 专家末步 | 专家最高 | best step |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['training_source']} | {r['bc_successes']}/64 | {r['stage_successes']}/64 | "
                     f"{r['single_successes']}/64 | {r['expert_successes']}/64 | {r['max']:.2%} | {r['best_step']} |")
    lines += ["", "![开发曲线](development_curves.png)", "", "BC虚线只是原BC水平参考，不代表它经过了本阶段PPO。", ""]
    for r in contrasts:
        lo, hi = r["descriptive_ci95"]
        lines.append(f"- {r['contrast']}：{100*r['difference']:+.2f}个百分点；来源与初态配对重采样描述性95%区间[{100*lo:+.2f}, {100*hi:+.2f}]。")
    lines += ["", "只有三个来源且开发集反复用于决策，区间不是独立确认或算法普适性的证明。",
        "中途最高不用于选择末步，六倍actor容量/计算开销仍保留；不是执行级残差。",
        "overview.csv保留每来源的新增/丢失成功与所有失败接触极值；episodes.csv含全部逐例结果。"]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
