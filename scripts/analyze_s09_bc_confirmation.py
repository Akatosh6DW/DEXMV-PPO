"""Matched-state, multi-initialization analysis of a frozen BC repair plan."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/s09-bc-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from retarget_s09_reference import digest

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"old_bc_ppo": "Old BC+PPO", "new_bc": "Repaired BC",
          "new_bc_ppo": "Repaired BC+PPO", "reference": "Reference replay"}


def bootstrap(matrix, other=None, paired_models=False, draws=4000):
    """Resample shared test states; pair training seeds only for BC->PPO.

    Old and new training seeds are independent replicates, not matched runs.
    Three training seeds are too few for strong population-level claims.
    """
    matrix = np.asarray(matrix, float)
    rng = np.random.default_rng(20261015)
    values = []
    for _ in range(draws):
        cases = rng.integers(matrix.shape[1], size=matrix.shape[1])
        models = rng.integers(len(matrix), size=len(matrix))
        value = matrix[np.ix_(models, cases)].mean()
        if other is not None:
            if paired_models:
                if other.shape != matrix.shape:
                    raise ValueError("Paired training runs require equal shape/order")
                other_models = models
            else:
                other_models = rng.integers(len(other), size=len(other))
            value -= other[np.ix_(other_models, cases)].mean()
        values.append(value)
    return np.quantile(values, [.025, .975]).tolist()


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    for path, expected in plan["execution_sources"].items():
        assert digest(ROOT/path) == expected, "Changed frozen evaluation source: " + path
    assert [j["training_seed"] for j in plan["jobs"] if j["group"] == "new_bc"] == [
        j["training_seed"] for j in plan["jobs"] if j["group"] == "new_bc_ppo"]
    output = a.plan.parent / "report"
    output.mkdir(exist_ok=True)
    raw, summaries, matrices, paired_initial = [], [], {}, None
    for job in plan["jobs"]:
        run = ROOT / job["output"]
        cfg = json.loads((run / "run_config.json").read_text())
        assert cfg["steps"] == 0 and cfg["eval_start"] == plan["eval_start"]
        assert cfg["eval_count"] == plan["eval_count"] and cfg["orientation"] == "uniform"
        assert cfg["approach_weight"] == 2 and cfg["coverage_weight"] == 2
        assert cfg["model_sha256"] == plan["model_sha256"]
        assert cfg["reference_sha256"] == plan["reference_sha256"]
        if job.get("checkpoint"):
            assert cfg["checkpoint_sha256"] == job["checkpoint_sha256"] == digest(ROOT / job["checkpoint"])
        folder = run / "step_000000000"
        assert json.loads((folder / "native_replay_audit.json").read_text())["status"] == "verified"
        with np.load(folder / "initial_states.npz", allow_pickle=False) as z:
            np.testing.assert_array_equal(z["seeds"], np.arange(plan["eval_start"], plan["eval_start"] + plan["eval_count"]))
            if paired_initial is None:
                paired_initial = z["qpos"].copy()
            else:
                np.testing.assert_array_equal(z["qpos"], paired_initial)
        episodes = json.loads((folder / "episodes.json").read_text())
        assert len(episodes) == plan["eval_count"]
        assert [r["seed"] for r in episodes] == list(range(plan["eval_start"], plan["eval_start"] + plan["eval_count"]))
        for e in episodes:
            assert e["released_rope_force_n"] == 0 and e["residual_rms_rad"] == 0
            if e["success30"]:
                assert e["hold_sec"] >= 30 and e["final_valid"] and not e["forbidden"]
                assert e["max_released_penetration_m"] <= .003 and e["initial_penetration_m"] <= .001
            raw.append(dict(group=job["group"], training_seed=job["training_seed"], **e))
        matrices.setdefault(job["group"], []).append([e["success30"] for e in episodes])
    matrices = {key: np.asarray(value, float) for key, value in matrices.items()}
    for group, matrix in matrices.items():
        rows = [r for r in raw if r["group"] == group]
        good = [r for r in rows if r["success30"]]
        summaries.append(dict(group=group, success_rate=float(matrix.mean()),
            per_training_seed=matrix.mean(axis=1).tolist(), successes=int(matrix.sum()),
            policy_episode_pairs=int(matrix.size), unique_test_states=matrix.shape[1],
            descriptive_ci95=bootstrap(matrix),
            mean_best_hold_sec=float(np.mean([r["hold_sec"] for r in rows])),
            all_case_max_penetration_mm=1000*max(r["max_released_penetration_m"] for r in rows),
            all_case_peak_force_n=max(r["peak_force_n"] for r in rows),
            successful_case_peak_force_n=max((r["peak_force_n"] for r in good), default=None)))
    differences = []
    for baseline in ("old_bc_ppo", "new_bc", "reference"):
        x, y = matrices["new_bc_ppo"], matrices[baseline]
        differences.append(dict(contrast="new_bc_ppo minus " + baseline,
            difference=float(x.mean()-y.mean()),
            descriptive_ci95=bootstrap(x, y, paired_models=baseline == "new_bc")))
    transitions = []
    for i, (bc, ppo) in enumerate(zip(matrices["new_bc"], matrices["new_bc_ppo"])):
        transitions.append(dict(replicate=i, retained=int(np.sum((bc==1)&(ppo==1))),
            lost=int(np.sum((bc==1)&(ppo==0))), gained=int(np.sum((bc==0)&(ppo==1))),
            neither=int(np.sum((bc==0)&(ppo==0)))))
    write_csv(output/"episodes.csv", raw)
    write_csv(output/"bc_to_ppo_transitions.csv", transitions)
    result = dict(plan_sha256=digest(a.plan), summaries=summaries, differences=differences,
        transitions=transitions, initial_states_exactly_paired=True,
        uncertainty="Descriptive hierarchical bootstrap; only three training replicates. No hardware guarantee.")
    (output/"statistics.json").write_text(json.dumps(result, indent=2)+"\n")
    fig, ax = plt.subplots(figsize=(9,4.5), layout="constrained")
    colors = ["#777777", "#3c8c70", "#c15e57", "#4a86b7"]
    for i, row in enumerate(summaries):
        ax.bar(i, 100*row["success_rate"], color=colors[i], alpha=.7)
        lo, hi = np.asarray(row["descriptive_ci95"])*100
        ax.plot([i,i], [lo,hi], color="black", lw=1)
        ax.scatter([i]*len(row["per_training_seed"]), 100*np.asarray(row["per_training_seed"]), color="black", s=18)
    ax.set(xticks=range(len(summaries)), xticklabels=[LABELS[r["group"]] for r in summaries],
        ylim=(0,100), ylabel="Strict 30s success (%)", title="Untouched random-orientation test states (n=128)")
    fig.savefig(output/"heldout_success.png", dpi=160)
    plt.close(fig)
    lines = ["# BC+PPO修复：冻结候选后的保留测试", "",
        "相同128个随机初态。学习方法各三个训练种子；不是把384次执行当成384个独立训练。",
        "三秒撤除悬吊，随后连续合格抓持30秒并在40秒结束时仍合格。失败不筛除。", "",
        "| 方法 | 成功率 | 各训练种子 | 成功/执行 | 描述性95%区间 |", "|---|---:|---|---:|---|"]
    for r in summaries:
        lines.append(f"| {LABELS[r['group']]} | {100*r['success_rate']:.2f}% | " +
            " / ".join(f"{100*x:.2f}%" for x in r["per_training_seed"]) +
            f" | {r['successes']}/{r['policy_episode_pairs']} | " +
            "--".join(f"{100*x:.2f}%" for x in r["descriptive_ci95"]) + " |")
    lines += ["", "![保留测试](heldout_success.png)", "", "## 配对初态上的差值", ""]
    for r in differences:
        lines.append(f"- {r['contrast']}：{100*r['difference']:+.2f}个百分点；描述性区间 " +
            " 至 ".join(f"{100*x:+.2f}" for x in r["descriptive_ci95"]) + " 个百分点。")
    lines += ["", "## 不能混淆的事", "",
        "- 旧4.17%来自旧测试集；本表旧权重也在新128例上重新评估，以本表为直接比较。",
        "- 修复同时增加了初始朝向模式条件、BC输入课程和BC更新量，并降低探索幅度；不是单变量最终比较，也不是等总调参成本。",
        "- 修复仍是单网络完整目标控制。PPO阶段全部实时观测和权重可训练，无参考动作加法。",
        "- 新BC与接续PPO按相同训练种子配对；旧/新训练种子独立重采样，测试初态始终配对。",
        "- 区间是描述性bootstrap，仅三个训练种子，不作强显著性或所有朝向成功的保证。",
        "- 成功条件约束穿透，但未限定峰值接触力；仿真成功不等于可直接上硬件。",
        "- 本轮没有从新RGB视频重新提取。使用原处理后DexMV示范，专门隔离训练算法问题。", "",
        "原始数据：episodes.csv；配对得失：bc_to_ppo_transitions.csv；完整统计：statistics.json。"]
    (output/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
