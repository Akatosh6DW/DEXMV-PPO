"""Audit BC expert specialization separately from any subsequent PPO gain."""

import csv
import json
from pathlib import Path
import pickle

import jax
import numpy as np

from analyze_s09_bc_confirmation import write_csv
from analyze_s09_late_experts import transitions
from analyze_s09_ranked_recovery import BASE
from retarget_s09_reference import digest
from run_s09_mode_experts import ARCHITECTURE


def equal_tree(a, b):
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    for x,y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_array_equal(x, y)


def main():
    read = lambda p: json.loads(p.read_text())
    runs = [BASE/"B12_feedback_ranked_recovery", BASE/"B15_feedback_recovery_experts"]
    configs = [read(p/"run_config.json") for p in runs]
    ignored = {"output", "sources", "policy_architecture"}
    assert {k:v for k,v in configs[0].items() if k not in ignored} == {
        k:v for k,v in configs[1].items() if k not in ignored}
    assert configs[1]["policy_architecture"] == ARCHITECTURE
    assert configs[0]["steps"] == 0 and configs[0]["bc_updates"] == 4000
    for name in configs[0]["sources"]:
        assert configs[0]["sources"][name] == configs[1]["sources"][name], name
    bc = [read(p/"recovery_bc_config.json") for p in runs]
    assert bc[0] == bc[1]
    assert digest(Path(bc[0]["source_checkpoint"])) == bc[0]["source_checkpoint_sha256"]
    before_params = pickle.loads(Path(bc[0]["source_checkpoint"]).read_bytes())
    after_params = pickle.loads((runs[1]/"step_000000000/params.pkl").read_bytes())
    for i in (0,2):
        equal_tree(before_params[i], after_params[i])
    equal_tree(before_params[1], after_params[1]["experts"][4])
    metadata = read(runs[1]/"recovery_expert_config.json")
    assert metadata["early_mode_counts"][4] == metadata["late_mode_counts"][4] == 0
    assert metadata["total_actor_parameters"] == 6*metadata["original_actor_parameters"]
    original = read(BASE/"E101_bc_independent_covariance_control/step_000000000/episodes.json")
    rows, groups, episodes, evidence = [], [], [], {}
    for run, config in zip(runs, configs):
        for name, sha in config["sources"].items():
            assert digest(run/"sources"/name) == sha
        summary = read(run/"summary.json")
        assert summary["status"] == "completed" and summary["final_step"] == 0
        with (run/"metrics.csv").open() as stream:
            metrics = [{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
        assert len(metrics) == 1 and np.isfinite(list(metrics[0].values())).all()
        folder = run/"step_000000000"
        audit = read(folder/"native_replay_audit.json")
        assert audit["status"] == "verified" and audit["checkpoint_sha256"] == digest(folder/"params.pkl")
        assert audit["trace_sha256"] == digest(folder/"preview_trajectories.npz")
        with np.load(folder/"initial_states.npz") as a, np.load(runs[0]/"step_000000000/initial_states.npz") as b:
            for key in ("seeds", "qpos"):
                np.testing.assert_array_equal(a[key], b[key])
        after = read(folder/"episodes.json")
        episodes.append(after)
        assert len(after) == 64 and metrics[0]["success30"] == np.mean([r["success30"] for r in after])
        for r in after:
            assert r["residual_rms_rad"] == 0 and r["released_rope_force_n"] == 0
            if r["success30"]:
                assert r["hold_sec"] >= 30 and r["final_valid"] and not r["forbidden"]
                assert r["initial_penetration_m"] <= .001 and r["max_released_penetration_m"] <= .003
        losses = read(run/"bc_loss.json")
        assert losses[-1]["step"] == 4000
        rows.append(dict(run=run.name, successes=sum(r["success30"] for r in after), n=64,
            **summary["success30"], **transitions(original, after), hold_sec=metrics[0]["hold_sec"],
            max_depth_mm=metrics[0]["max_penetration_m"]*1000, peak_force_n=metrics[0]["peak_force_n"],
            last_train_loss=losses[-1]["train_loss"], last_validation_loss=losses[-1]["validation_loss"]))
        for mode in range(6):
            a = [r for r in original if r["selected_reference"] == mode]
            b = [r for r in after if r["selected_reference"] == mode]
            groups.append(dict(run=run.name, mode=mode, n=len(b), successes=sum(r["success30"] for r in b),
                **transitions(a,b)))
        evidence[run.name] = {name:digest(run/name) for name in ("run_config.json", "metrics.csv", "summary.json",
            "step_000000000/episodes.json", "step_000000000/native_replay_audit.json")}
    paired = transitions(*episodes)
    out = BASE/"report/recovery_experts"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/"overview.csv", rows)
    write_csv(out/"orientation.csv", groups)
    write_csv(out/"paired_episodes.csv", [dict(seed=a["seed"], mode=a["selected_reference"],
        flat=a["success30"], expert=b["success30"], flat_hold=a["hold_sec"], expert_hold=b["hold_sec"])
        for a,b in zip(*episodes)])
    (out/"evidence.json").write_text(json.dumps(dict(status="verified_development_only", evidence=evidence,
        paired=paired, critic_normalizer_unchanged=True, unsupervised_mode4_unchanged=True,
        source_sha256=digest(Path(__file__)), scope="One BC seed, reused development states, sixfold actor capacity, zero PPO"), indent=2)+"\n")
    lines = ["# 分朝向恢复BC：B15对照B12", "",
        "同I27源、D45数据、4000监督更新、同采样随机数，0步PPO。B15增加六倍actor参数。",
        "原BC为26/64；第4类训练/验证无样本，专家参数与原BC逐元素相同，仍计入全部开发评估。", "",
        "| 组 | 成功 | 相对原BC新增/丢失 | 平均合格秒 | 最大穿透mm | 峰值N |",
        "|---|---:|---:|---:|---:|---:|", *[
        f"| {r['run']} | {r['successes']}/64 | {r['gained']}/{r['lost']} | {r['hold_sec']:.3f} | {r['max_depth_mm']:.3f} | {r['peak_force_n']:.3f} |" for r in rows],
        "", f"专家相对单网络新增{paired['gained']}例、丢失{paired['lost']}例。",
        "first=last=max仅指唯一训练后评估，best_step0是PPO步数。不是成功率学习曲线。",
        "保存预览动作和独立原生回放已核验；接触极值包含失败。不是新初态确认或多训练重复。",
        "分专家同时增加容量、改变参数共享及梯度分配，不能只凭该对照证明唯一干扰机制。"]
    (out/"results.md").write_text("\n".join(lines)+"\n")
    diagnostic = []
    for name in ("D48_expert_student_closed", "D48_expert_student_forced", "D49_portfolio_recovery_teacher"):
        folder = BASE/name
        summary = read(folder/"summary.json")
        assert summary["status"] == "completed" and summary["cases"] == 2
        assert summary["teacher_collection_sha256"] == digest(BASE/"D44_teacher_full_pool/results.json")
        assert summary["source_sha256"] == digest(folder/"probe_s09_recovery_student.py")
        checkpoint = (runs[1]/"step_000000000/params.pkl" if name.startswith("D48") else
                      BASE/"confirmation_portfolio_20261019/candidate/params.pkl")
        for r in read(folder/"results.json"):
            assert r["checkpoint_sha256"] == digest(checkpoint)
            assert r["trace_sha256"] == digest(folder/f"{r['run']}_seed{r['seed']}.npz")
            assert r["native_replay_max_qpos_error"] < 3e-6 and r["native_replay_max_qvel_error"] < 1e-4
            diagnostic.append(dict(diagnostic=name, **r))
    write_csv(out/"recovery_diagnostics.csv", diagnostic)
    (out/"recovery_diagnostics.md").write_text("\n".join([
        "# 分专家学生与已有闭环老师：相同前缀恢复", "",
        "D48检查B15；D49检查已训练的历史PPO组合。都是D44两条已见训练状态，不是新测试。",
        "forced使用老师观测，仅作诊断，不能部署；全部6条执行轨迹原生回放核验通过。", "",
        "| 诊断 | 种子 | 合格秒 | 严格成功 | 峰值N |",
        "|---|---:|---:|---|---:|",
        *[f"| {r['diagnostic']} | {r['seed']} | {r['hold_sec']:.2f} | {r['success30']} | {r['peak_force_n']:.3f} |" for r in diagnostic],
        "", "D49成功例说明已有闭环策略可提供一例恢复，但还有一个失败，不能当万能老师。",
        "该老师继承1310720在线步和既有开发集选检查点成本，不是纯视频数据或免费专家。",
    ])+"\n")
    print(json.dumps(dict(rows=rows, paired=paired)), flush=True)


if __name__ == "__main__":
    main()
