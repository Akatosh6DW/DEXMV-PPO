"""Confirm a fixed historical-checkpoint portfolio on new paired initials."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from analyze_s09_bc_confirmation import bootstrap, write_csv, plt
from analyze_s09_expert_confirmation import holm
from analyze_s09_ppo_confirmation import transitions, failure_gate_summary
from confirm_s09_late_experts import ROOT, BASE
from retarget_s09_reference import digest


def main():
    out = BASE/"confirmation_portfolio_20261019"
    read = lambda p:json.loads(p.read_text())
    plan, selection = [read(out/name) for name in ("plan.json", "selection.json")]
    assert digest(out/"selection.json") == plan["selection_sha256"]
    assert (plan["eval_start"],plan["eval_count"]) == (180000,256)
    assert selection["extra_ppo_steps"] == 0
    for path,value in plan["execution_sources"].items():
        assert digest(ROOT/path) == value, path
    for source in selection["sources"]:
        path = ROOT/source["checkpoint"]
        assert digest(path) == source["checkpoint_sha256"]
        assert digest(path.parent/"episodes.json") == source["episodes_sha256"]
    initial = None
    summaries, episodes, modes, gates, matrices, evidence = [], [], [], [], {}, {}
    for job in plan["jobs"]:
        run = ROOT/job["output"]
        cfg, summary = [read(run/name) for name in ("run_config.json", "summary.json")]
        assert summary["status"] == "completed" and summary["final_step"] == cfg["steps"] == 0
        for key in ("eval_start", "eval_count", "env_config", "model_sha256", "reference_sha256"):
            assert cfg[key] == plan[key], key
        assert cfg["checkpoint_sha256"] == job["checkpoint_sha256"] == digest(ROOT/job["checkpoint"])
        assert read(run/"quality_config.json")["primary_success_unchanged"]
        assert read(run/"failure_config.json")["original_success_unchanged"]
        folder = run/"step_000000000"
        audit = read(folder/"native_replay_audit.json")
        assert audit["status"] == "verified" and audit["normalized_action_tolerance"] == 1e-5
        assert audit["checkpoint_sha256"] == digest(folder/"params.pkl")
        assert audit["trace_sha256"] == digest(folder/"preview_trajectories.npz")
        for path,value in audit.get("supplementary_audit_sources",{}).items():
            assert digest(ROOT/path) == value
        with np.load(folder/"initial_states.npz") as z:
            np.testing.assert_array_equal(z["seeds"], np.arange(180000,180256))
            if initial is None:
                initial = z["qpos"].copy()
            np.testing.assert_array_equal(z["qpos"], initial)
        rows = read(folder/"episodes.json")
        assert [r["seed"] for r in rows] == list(range(180000,180256))
        for r in rows:
            assert r["released_rope_force_n"] == r["residual_rms_rad"] == 0
            if r["success30"]:
                assert r["hold_sec"] >= 30 and r["final_valid"] and not r["forbidden"]
                assert r["max_released_penetration_m"] <= .003 and r["initial_penetration_m"] <= .001
        array = np.array([[r["success30"] for r in rows]],float)
        assert array.mean() == summary["rows"][0]["success30"]
        matrices[job["name"]] = array
        ci = binomtest(int(array.sum()),array.size).proportion_ci(method="wilson")
        summaries.append(dict(group=job["name"],successes=int(array.sum()),n=array.size,
            success_rate=float(array.mean()),ci95=[float(ci.low),float(ci.high)],
            hold_sec=float(np.mean([r["hold_sec"] for r in rows])),
            max_released_depth_mm=1000*max(r["max_released_penetration_m"] for r in rows),
            max_all_depth_mm=1000*max(r["max_all_penetration_m"] for r in rows),
            peak_force_n=max(r["peak_force_n"] for r in rows),
            successful_peak_force_n=max((r["peak_force_n"] for r in rows if r["success30"]),default=None),
            all_depth_success_rate=float(np.mean([r["success30_all_depth_le_3mm"] for r in rows])),
            path_online_steps=job["path_online_steps"],actor_parameter_multiplier=job["actor_parameter_multiplier"]))
        for mode in range(6):
            part = [r for r in rows if r["selected_reference"] == mode]
            modes.append(dict(group=job["name"],mode=mode,n=len(part),successes=sum(r["success30"] for r in part)))
        gates.extend(dict(group=job["name"],**r) for r in failure_gate_summary(rows))
        episodes.extend(dict(group=job["name"],**r) for r in rows)
        evidence[job["name"]] = dict(episodes_sha256=digest(folder/"episodes.json"),audit_sha256=digest(folder/"native_replay_audit.json"),
            supplementary_audit_sources=audit.get("supplementary_audit_sources",{}))
    paired = []
    for name in ("bc","expert"):
        x,y = matrices["portfolio"],matrices[name]
        delta = transitions(y,x)
        different = delta["gained"]+delta["lost"]
        p = binomtest(delta["gained"],different).pvalue if different else 1.
        paired.append(dict(contrast="portfolio minus "+name,difference=float((x-y).mean()),
            ci95=bootstrap(x,y,paired_models=True,draws=20000),exact_mcnemar_p=float(p),**delta))
    for r,p in zip(paired,holm([r["exact_mcnemar_p"] for r in paired])):
        r["holm_p_two_comparisons"] = p
    output = out/"report"
    output.mkdir(exist_ok=True)
    result = dict(status="completed_checkpoint_selection_confirmation",plan_sha256=digest(out/"plan.json"),
        summaries=summaries,paired=paired,selection=selection,evidence=evidence,source_sha256=digest(Path(__file__)),
        limitations=plan["limitations"],interval_scope="Fixed candidate policies and random initial states; not training-seed uncertainty")
    (output/"statistics.json").write_text(json.dumps(result,indent=2)+"\n")
    (output/"episodes.json").write_text(json.dumps(episodes,indent=2)+"\n")
    for name,rows in (("summary.csv",summaries),("paired.csv",paired),("orientation.csv",modes),("failure_gates.csv",gates)):
        write_csv(output/name,rows)
    write_csv(output/"episodes.csv",[dict(r,failure_gates=json.dumps(r["failure_gates"],sort_keys=True)) for r in episodes])
    fig,ax = plt.subplots(figsize=(7,4.5),layout="constrained")
    for i,r in enumerate(summaries):
        ax.bar(i,100*r["success_rate"],color=["#777777","#a94b55","#418b91"][i])
        ax.plot([i,i],100*np.array(r["ci95"]),color="black")
    ax.set(xticks=range(3),xticklabels=["BC","Expert endpoint","Checkpoint portfolio"],
        ylim=(0,100),ylabel="Strict success (%)",title="256 new paired initial states; no new training")
    fig.savefig(output/"success.png",dpi=160)
    plt.close(fig)
    lines = ["# 历史检查点组合：新的256初态确认", "",
        "此组新增0步PPO，是按旧开发初态选择历史专家；继承同一训练路径1310720步。不是又训练出一个更强的PPO，也不是残差。", "",
        "| 组 | 成功/256 | 成功率 | 平均合格秒 | 释放后最大穿透mm | 成功案例峰值N |", "|---|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['group']} | {r['successes']} | {r['success_rate']:.2%} | {r['hold_sec']:.3f} | {r['max_released_depth_mm']:.3f} | {r['successful_peak_force_n']} |")
    lines += ["", "| 组合减对照 | 净收益 | 配对95%区间 | 新增/丢失 | Holm p |", "|---|---:|---:|---:|---:|"]
    for r in paired:
        lines.append(f"| {r['contrast']} | {r['difference']:.2%} | [{r['ci95'][0]:.2%},{r['ci95'][1]:.2%}] | {r['gained']}/{r['lost']} | {r['holm_p_two_comparisons']:.6g} |")
    lines += ["", "## 限制与复现", "",
        "固定候选后只跑180000--180255；此前170000确认用于分析，不混用为本组未见测试。配对bootstrap20000次，McNemar双侧并对两个比较做Holm校正。区间不涵盖训练种子或开发集选择的不确定性。",
        "源池固定三时点、六个类别，每类按旧开发成功数选择、平局取早期。不能据本轮测试标签重选后仍称独立确认。",
        "全部失败保留，所有案例记录质量门控；保存预览绑定实际权重并原生回放，不声称全例另存重放。原30秒严格条件未放宽。",
        "组合及E78各自的预案大批量动作审计均有1个元素误差1.02371e-5，略超1e-5；首次失败日志保留。两组均后补逐帧同批量形状审计，不改原冻结计划、动作、权重、物理或容差。补充审计源码哈希见statistics.json，不能称为全部按原审计预案直接通过。",
        "与原BC相比，组合actor六倍参数，附加开发选择；与E78相比参数量相同。新增/丢失列展示仍未完全消失的遗忘。原始逐例JSON/CSV、分朝向与失效表、来源哈希在本目录。"]
    (output/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
