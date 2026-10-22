"""Analyze ALL paired continuation branches; select candidates on development only."""

import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/s09-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_s09_study import CHINESE, interval
from retarget_s09_reference import digest
from run_s09_reward_study import DEST, METHODS, OUT, ROOT, STRATA, VERSIONS, protocol


def train_run(method, version):
    return OUT / "train" / f"{method}_s9101" if version == "original" else DEST / "train" / f"{version}_{method}"


def main():
    plan = protocol()
    candidates = []
    training = []
    for method in METHODS:
        choices = []
        for order, version in enumerate(VERSIONS):
            run = train_run(method, version)
            summary = json.loads((run / "summary.json").read_text())
            config = json.loads((run / "run_config.json").read_text())
            step = 163840 if version == "original" else 40960
            assert summary["status"] == "completed" and summary["final_step"] == step
            if version != "original":
                assert config["checkpoint_sha256"] == plan["checkpoints"][method]
                assert config["seed"] == 10101 and config["steps"] == 40960
                assert config["learning_rate"] == plan["recipes"][method]["learning_rate"]
                expected_reward = "S09-R1_penetration" if version == "penalty_R1" else "S09-R0"
                assert config["env_config"]["reward_version"] == expected_reward
            with (run / "metrics.csv").open() as file:
                metrics = list(csv.DictReader(file))
            if version != "original":
                parent = json.loads((train_run(method, "original") / "summary.json").read_text())
                for key in ("success30", "hold_sec"):
                    np.testing.assert_allclose(float(metrics[0][key]), parent[key]["last"], atol=1e-6)
            last = {k: float(v) for k, v in metrics[-1].items()}
            assert last["step"] == step
            params = run / f"step_{step:09d}/params.pkl"
            item = dict(method=method, version=version, params=str(params.relative_to(ROOT)),
                params_sha256=digest(params), success30=summary["success30"],
                hold_sec=summary["hold_sec"], last_metrics=last)
            training.append(item)
            key = (last["success30"], -last["max_penetration_m"], last["hold_sec"], -order)
            choices.append((key, item))
        # This selection deliberately happens before loading held-out results.
        selected = max(choices, key=lambda pair: pair[0])[1]
        candidates.append(dict(**selected,
            selection="Development final success desc, penetration asc, hold desc, original preferred on tie",
            actor_dim=123, action_dim=17, control_dt=.05,
            model=str((OUT / "inputs/model.xml").relative_to(ROOT)),
            model_sha256=digest(OUT / "inputs/model.xml"),
            reference=str((OUT / "inputs/reference.npz").relative_to(ROOT)) if method == "residual" else None,
            reference_sha256=digest(OUT / "inputs/reference.npz") if method == "residual" else None,
            action="clip(reference + .08 * clip(policy,-1,1),limits), then .1rad slew" if method == "residual"
                else "normalized policy mapped to joint limits, then .1rad slew"))
    rows, raw, matrices, audits = [], [], {}, []
    for orientation, (start, count) in STRATA.items():
        initial = None
        for method in (*METHODS, "reference"):
            for version in (("original",) if method == "reference" else VERSIONS):
                name = "reference" if method == "reference" else f"{version}_{method}"
                run = DEST / "eval" / orientation / name
                summary = json.loads((run / "summary.json").read_text())
                config = json.loads((run / "run_config.json").read_text())
                assert summary["status"] == "completed" and summary["final_step"] == 0
                assert config["orientation"] == orientation and config["eval_start"] == start and config["eval_count"] == count
                assert config["model_sha256"] == digest(OUT / "inputs/model.xml")
                assert config["reference_sha256"] == digest(OUT / "inputs/reference.npz")
                if method != "reference":
                    step = 163840 if version == "original" else 40960
                    assert config["checkpoint_sha256"] == digest(train_run(method, version) / f"step_{step:09d}/params.pkl")
                folder = run / "step_000000000"
                with np.load(folder / "initial_states.npz") as data:
                    np.testing.assert_array_equal(data["seeds"], np.arange(start, start + count))
                    if initial is None:
                        initial = data["qpos"].copy()
                    else:
                        np.testing.assert_array_equal(data["qpos"], initial)
                episodes = json.loads((folder / "episodes.json").read_text())
                assert len(episodes) == count and [e["seed"] for e in episodes] == list(range(start, start + count))
                for e in episodes:
                    assert e["released_rope_force_n"] == 0
                    if e["success30"]:
                        assert e["hold_sec"] >= 30 and e["final_valid"] and not e["forbidden"]
                        assert e["initial_penetration_m"] <= .001 and e["max_released_penetration_m"] <= .003
                    raw.append(dict(orientation=orientation, method=method, version=version, **e))
                matrix = np.array([[e["success30"] for e in episodes]], float)
                matrices[orientation, method, version] = matrix
                rows.append(dict(orientation=orientation, method=method, version=version,
                    successes=int(matrix.sum()), count=count, success_rate=float(matrix.mean()),
                    hold_sec=float(np.mean([e["hold_sec"] for e in episodes])),
                    max_penetration_mm=max(e["max_released_penetration_m"] for e in episodes)*1000,
                    peak_force_n=max(e["peak_force_n"] for e in episodes)))
                audit = json.loads((folder / "native_replay_audit.json").read_text())
                assert audit["status"] == "verified"
                audits.append(audit)
    assert len(raw) == 480 and len(rows) == 20 and len(audits) == 20
    differences = []
    for orientation in STRATA:
        for method in METHODS:
            delta = matrices[orientation, method, "penalty_R1"] - matrices[orientation, method, "continued_R0"]
            differences.append(dict(orientation=orientation, method=method,
                contrast="penalty_R1 minus continued_R0", difference=float(delta.mean()),
                episode_only_ci95=interval(delta)))
    report = DEST / "report"
    report.mkdir(exist_ok=True)
    for name, records in (("episodes", raw), ("summary", rows)):
        with (report / (name + ".csv")).open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        (report / (name + ".json")).write_text(json.dumps(records, indent=2) + "\n")
    for name, data in (("training", training), ("paired_differences", differences),
                       ("candidate_policies", candidates), ("replay_audits", audits)):
        (report / (name + ".json")).write_text(json.dumps(data, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, orientation in zip(axes, STRATA):
        for j, version in enumerate(VERSIONS):
            values = [next(r["success_rate"] for r in rows if r["orientation"] == orientation
                and r["method"] == method and r["version"] == version) * 100 for method in METHODS]
            ax.bar(np.arange(3) + (j-1)*.24, values, .24, label=version)
        reference = next(r["success_rate"] for r in rows if r["orientation"] == orientation and r["method"] == "reference")
        ax.axhline(reference*100, color="black", linestyle=":", label="Reference")
        ax.set(xticks=np.arange(3), xticklabels=["PPO", "BC+PPO", "Residual"],
            ylabel="Strict 30s success (%)", ylim=(0, 100), title=orientation)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=8)
    fig.savefig(report / "reward_comparison.png", dpi=150)
    plt.close(fig)
    lines = ["# S09奖励迭代：单种子探索性结果", "",
        "固定R0比较完成后，三种学习方法各从seed9101末步出发。两个续训分支都增加40960步、同样重置优化器、使用相同新种子10101，区别只在穿透惩罚。",
        "原R0=163840步；同奖励续训/R1=累计204800步。因此应比较两种续训分支来判断惩罚作用，不能把全部增益解释为奖励改善。",
        "新保留初态：随机朝向32个、虎口方向16个，总计480回合；只有一个续训种子，不能替代主实验的3种子统计。", "",
        "| 朝向 | 方法 | 版本 | 成功/次数 | 成功率 | 平均最长保持(s) | 最大穿透(mm) |", "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['orientation']} | {CHINESE[r['method']]} | {r['version']} | {r['successes']}/{r['count']} | "
            f"{r['success_rate']*100:.2f}% | {r['hold_sec']:.2f} | {r['max_penetration_mm']:.3f} |")
    lines += ["", "![奖励对比](reward_comparison.png)", "", "## 开发集选择的候选", "",
        "候选选择只看开发集末步：成功率优先，再比较穿透、保持时长、少训练优先。没有根据本页测试成绩选模型。"]
    lines.append("这里的候选仅限seed9101原模型与其续训分支，不是从全工程全部种子中寻找最优权重；原R0另外两个种子仍保留。")
    for item in candidates:
        lines.append(f"- {CHINESE[item['method']]}：{item['version']}；开发成功率{item['last_metrics']['success30']*100:.2f}%；权重位于主工程`{item['params']}`。")
    lines += ["", "参考回放没有PPO奖励，不做续训；它仍在新保留初态上评估。",
        "惩罚既可能改善接触，也可能让手不敢收拢；若成功率下降，不应只拿力峰值下降声称策略改善。候选权重不是硬件安全或全方向成功的保证。",
        "所有20项评估保存的预览均通过原生回放核验；未声称全部480条轨迹独立重放。配对区间仅反映测试初态抽样，不能估计训练随机性。",
        "", "- [原始逐回合数据](episodes.csv)", "- [续训first/last/max/best_step](training.json)",
        "- [相同追加步数的配对差值](paired_differences.json)", "- [候选策略及依赖清单](candidate_policies.json)"]
    (report / "results.md").write_text("\n".join(lines) + "\n")
    (report / "verification.json").write_text(json.dumps(dict(status="verified", evaluations=20, episodes=480,
        preview_audits=20, protocol_sha256=digest(DEST / "protocol.json")), indent=2) + "\n")
    print(json.dumps(dict(output=str(report), episodes=len(raw), candidates=[(r["method"], r["version"]) for r in candidates])), flush=True)


if __name__ == "__main__":
    main()
