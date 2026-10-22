"""Separate portfolio initialization gains from subsequent protected PPO gains."""

import json
from pathlib import Path

import numpy as np

from analyze_s09_bc_confirmation import write_csv
from analyze_s09_late_experts import BASE, transitions
from analyze_s09_angular_feedback import paired_initials
from retarget_s09_reference import digest


def main():
    read = lambda p:json.loads(p.read_text())
    runs = dict(protected=BASE/"E83_portfolio_protected",unprotected=BASE/"E84_portfolio_unprotected")
    configs = {k:read(p/"run_config.json") for k,p in runs.items()}
    for key in ("checkpoint_sha256", "model_sha256", "reference_sha256", "env_config", "seed", "std",
                "steps", "evals", "eval_start", "eval_count", "learning_rate", "eval_batch", "backend", "policy_architecture"):
        assert configs["protected"][key] == configs["unprotected"][key], key
    assert configs["protected"]["steps"] == 40960 and configs["protected"]["evals"] == 2
    for filename in configs["protected"]["sources"].keys() & configs["unprotected"]["sources"].keys():
        assert configs["protected"]["sources"][filename] == configs["unprotected"]["sources"][filename],filename
    a,b = [read(p/"constraint_config.json") for p in runs.values()]
    for key in ("actor_learning_rate", "critic_learning_rate", "teacher_kl_weight"):
        assert a[key] == b[key], key
    a,b = [read(p/"improvement_config.json") for p in runs.values()]
    for key in ("training_overrides", "initial_std_override"):
        assert a[key] == b[key],key
    protection = read(runs["protected"]/"protection_config.json")
    frozen = protection["protected_modes"]
    assert frozen == [0,1,3]
    parameter_audit = read(runs["protected"]/"protection_audit.json")
    assert parameter_audit["status"] == "verified"
    assert parameter_audit["initial_checkpoint_sha256"] == digest(runs["protected"]/"step_000000000/params.pkl")
    assert parameter_audit["final_checkpoint_sha256"] == digest(runs["protected"]/"step_000040960/params.pkl")
    for row in parameter_audit["experts"]:
        assert row["protected"] == (row["mode"] in frozen)
        if row["protected"]:
            assert row["changed_parameters"] == 0
    rows,paired,modes,raw,evidence,eps_by_run = [],[],[],[],{},{}
    summaries = {}
    initial = None
    for name,run in runs.items():
        summary = read(run/"summary.json")
        assert summary["status"] == "completed" and summary["final_step"] == 40960
        assert [r["step"] for r in summary["rows"]] == [0,40960]
        summaries[name] = {k:summary[k] for k in ("success30","hold_sec")}
        eps_by_run[name] = {}
        for metric in summary["rows"]:
            step = metric["step"]
            folder = run/f"step_{step:09d}"
            with np.load(folder/"initial_states.npz") as z:
                if initial is None:
                    initial = dict(z)
                paired_initials(initial,z)
            eps = read(folder/"episodes.json")
            assert len(eps) == 64 and np.mean([r["success30"] for r in eps]) == metric["success30"]
            audit = read(folder/"native_replay_audit.json")
            assert audit["status"] == "verified" and audit["normalized_action_tolerance"] == 1e-5
            assert audit["checkpoint_sha256"] == digest(folder/"params.pkl")
            assert audit["trace_sha256"] == digest(folder/"preview_trajectories.npz")
            for path,value in audit.get("supplementary_audit_sources",{}).items():
                assert digest(Path(__file__).resolve().parents[1]/path) == value
            eps_by_run[name][step] = eps
            rows.append(dict(group=name,step=step,success30=metric["success30"],hold_sec=metric["hold_sec"],
                max_depth_mm=1000*metric["max_penetration_m"],peak_force_n=metric["peak_force_n"],
                forbidden_rate=metric["forbidden_rate"],released_rope_force_n=metric["released_rope_force_n"]))
            raw.extend(dict(group=name,step=step,**r) for r in eps)
            evidence[f"{name}_{step}"] = dict(episodes_sha256=digest(folder/"episodes.json"),audit_sha256=digest(folder/"native_replay_audit.json"))
            for mode in range(6):
                part = [r for r in eps if r["selected_reference"] == mode]
                modes.append(dict(group=name,step=step,mode=mode,n=len(part),successes=sum(r["success30"] for r in part)))
        paired.append(dict(contrast=name+" final minus own start",**transitions(eps_by_run[name][0],eps_by_run[name][40960])))
    assert eps_by_run["protected"][0] == eps_by_run["unprotected"][0], "Paired starting policies differ"
    protected_rows = {step:[r for r in eps if r["selected_reference"] in frozen]
                      for step,eps in eps_by_run["protected"].items()}
    # Frozen actors must retain actual deterministic behavior, not just counts.
    assert protected_rows[0] == protected_rows[40960], "Frozen-pose physical behavior changed"
    paired.append(dict(contrast="protected final minus unprotected final",
        **transitions(eps_by_run["unprotected"][40960],eps_by_run["protected"][40960])))
    output = BASE/"report/protected_portfolio"
    output.mkdir(parents=True,exist_ok=True)
    result = dict(status="completed_development_only",summaries=summaries,paired=paired,
        frozen_pose_episode_fields_identical=True,protected_case_count=len(protected_rows[0]),
        parameter_audit=parameter_audit,evidence=evidence,rows=rows,source_sha256=digest(Path(__file__)),
        limitations="One new PPO seed, reused development set; inherited 1310720 online steps and prior checkpoint selection, not independent test")
    (output/"statistics.json").write_text(json.dumps(result,indent=2)+"\n")
    for name,values in (("curve.csv",rows),("paired.csv",paired),("orientation.csv",modes),("episodes.csv",raw)):
        write_csv(output/name,values)
    lines = ["# E83/E84：保护早期强项后再用PPO", "",
        "两组从相同组合权重开始，新增40960步、相同seed和奖励。保护组冻结0/1/3类actor，不保护组全部更新；所有朝向交互仍计预算。", "",
        "| 组 | 追加步 | 成功率 | 平均合格秒 | 最大穿透mm | 峰值N |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['group']} | {r['step']} | {r['success30']:.2%} | {r['hold_sec']:.3f} | {r['max_depth_mm']:.3f} | {r['peak_force_n']:.3f} |")
    lines += ["", f"first/last/max/best_step：{summaries}", "", "| 比较 | 新增 | 丢失 |", "|---|---:|---:|"]
    for r in paired:
        lines.append(f"| {r['contrast']} | {r['gained']} | {r['lost']} |")
    lines += ["", f"冻结类别共{len(protected_rows[0])}个开发案例，起末逐例全部字段一致，参数也逐元素不变。",
        "保存预览的动作/物理审计通过，最大穿透包含失败案例。不能把组合起点相对原BC已有的提升归因于本轮PPO。",
        "单种子、64开发初态，不是新的保留测试；参数保护不保证未保护类别也改善，更不保证原先失败的冻结类别会自愈。"]
    (output/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
