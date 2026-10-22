"""Analyze a frozen, matched-state BC/PPO and reference/residual confirmation."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_s09_bc_confirmation import bootstrap, write_csv
import matplotlib.pyplot as plt
from retarget_s09_reference import digest
from run_s09_study import ROOT, check_protocol
from freeze_s09_ppo_confirmation import HELDOUT_START, HELDOUT_COUNT


LABELS={"bc":"仅BC", "bc_ppo":"BC＋PPO（开发集选定）",
        "bc_ppo_last":"BC＋PPO（训练末步）",
        "reference":"零修正参考回放", "residual":"参考＋残差PPO"}


def plot_results(summaries, differences, output):
    """Show every training repeat and late endpoints, without a cropped rate axis."""
    labels={"bc":"BC", "bc_ppo":"BC + PPO\nselected",
            "bc_ppo_last":"BC + PPO\nlast", "reference":"Reference", "residual":"Residual PPO"}
    colors={"bc":"#707070", "bc_ppo":"#087e8b", "bc_ppo_last":"#79a9a1",
            "reference":"#b07b24", "residual":"#ae4946"}
    fig, ax=plt.subplots(figsize=(10,5),layout="constrained")
    for i,row in enumerate(summaries):
        ax.bar(i,100*row["success_rate"],color=colors[row["group"]],alpha=.75)
        lo,hi=np.asarray(row["descriptive_ci95"])*100
        ax.plot([i,i],[lo,hi],color="black",lw=1)
        ax.scatter(i+np.linspace(-.08,.08,len(row["per_training_seed"])),
            100*np.asarray(row["per_training_seed"]),color="black",s=22,zorder=3)
        ax.plot([i-.22,i+.22],[100*row["secondary_all_depth_success_rate"]]*2,
            color="white",lw=2)
    ax.set(xticks=range(len(summaries)),xticklabels=[labels[r["group"]] for r in summaries],
        ylim=(0,100),ylabel="Strict success (%)",
        title=f"Frozen candidates: {HELDOUT_COUNT} shared heldout initial states")
    ax.text(.01,.98,"Dots: training repeats; lines: descriptive 95% intervals\nWhite mark: additional whole-episode depth <= 3 mm", 
        transform=ax.transAxes,va="top",fontsize=9)
    fig.savefig(output/"heldout_success.png",dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5),layout="constrained")
    for i,row in enumerate(differences):
        lo,hi=np.asarray(row["descriptive_ci95"])*100
        ax.plot([lo,hi],[i,i],color="#087e8b",lw=2)
        ax.scatter(100*row["difference"],i,color="#ae4946",zorder=3)
    ax.axvline(0,color="black",linestyle="--",lw=1)
    ax.set(yticks=range(len(differences)),
        yticklabels=[r["contrast"].replace("bc_ppo_last","BC+PPO last").replace("bc_ppo","BC+PPO selected")
                    .replace("residual","Residual PPO").replace("reference","Reference").replace(" minus bc"," minus BC") for r in differences],
        xlabel="Paired success difference (percentage points)",
        title="Descriptive 95% intervals; only three independent training repeats")
    ax.invert_yaxis()
    ax.grid(axis="x",alpha=.2)
    fig.savefig(output/"paired_gains.png",dpi=160)
    plt.close(fig)


def transitions(before,after):
    before,after=np.asarray(before,bool),np.asarray(after,bool)
    if before.shape!=after.shape:
        raise ValueError("Pair identical cases")
    return dict(retained=int(np.sum(before&after)),lost=int(np.sum(before&~after)),
        gained=int(np.sum(~before&after)),neither=int(np.sum(~before&~after)))


def failure_gate_summary(episodes):
    """Nonexclusive diagnostic counts, not a causal partition of failures."""
    if not episodes:
        raise ValueError("Cannot summarize an empty evaluation")
    gates=set(episodes[0]["failure_gates"]["invalid_seconds_by_gate"])
    for episode in episodes:
        diagnostic=episode["failure_gates"]
        if diagnostic["success30"] != episode["success30"]:
            raise ValueError("Failure diagnostics must preserve the primary label")
        if set(diagnostic["invalid_seconds_by_gate"]) != gates or set(diagnostic["break_events_by_gate"]) != gates:
            raise ValueError("Every episode must contain the same diagnostic gates")
    failed=[e["failure_gates"] for e in episodes if not e["success30"]]
    return [dict(gate=gate,failed_policy_episode_pairs=len(failed),
        failed_pairs_with_gate=sum(e["invalid_seconds_by_gate"][gate]>0 for e in failed),
        invalid_seconds_total=sum(e["invalid_seconds_by_gate"][gate] for e in failed),
        break_events_total=sum(e["break_events_by_gate"][gate] for e in failed)) for gate in sorted(gates)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    check_protocol()
    for path,expected in plan["execution_sources"].items():
        if digest(ROOT/path)!=expected:
            raise ValueError("Changed frozen execution source: "+path)
    if (plan["eval_start"],plan["eval_count"])!=(HELDOUT_START,HELDOUT_COUNT):
        raise ValueError("Use declared heldout cases")
    groups={key:[] for key in LABELS}
    raw=[]
    initial=None
    expected_seeds=list(range(HELDOUT_START,HELDOUT_START+HELDOUT_COUNT))
    for job in plan["jobs"]:
        run=ROOT/job["output"]
        cfg=json.loads((run/"run_config.json").read_text())
        summary=json.loads((run/"summary.json").read_text())
        assert summary["status"]=="completed" and summary["final_step"]==0
        assert cfg["steps"]==0 and cfg["eval_start"]==HELDOUT_START and cfg["eval_count"]==HELDOUT_COUNT
        assert cfg["orientation"]=="uniform" and cfg["approach_weight"]==cfg["coverage_weight"]==2
        assert cfg["model_sha256"]==plan["model_sha256"] and cfg["reference_sha256"]==plan["reference_sha256"]
        if job.get("checkpoint"):
            assert cfg["checkpoint_sha256"]==job["checkpoint_sha256"]==digest(ROOT/job["checkpoint"])
        folder=run/"step_000000000"
        audit=json.loads((folder/"improvement_audit.json").read_text())
        assert audit["status"]=="verified" and audit["trace_sha256"]==digest(folder/"preview_trajectories.npz")
        assert audit["native_audit_sha256"]==digest(folder/"native_replay_audit.json")
        assert json.loads((run/"quality_config.json").read_text())["primary_success_unchanged"]
        assert json.loads((run/"failure_config.json").read_text())["original_success_unchanged"]
        with np.load(folder/"initial_states.npz",allow_pickle=False) as data:
            np.testing.assert_array_equal(data["seeds"],expected_seeds)
            if initial is None:
                initial=data["qpos"].copy()
            else:
                np.testing.assert_array_equal(initial,data["qpos"])
        episodes=json.loads((folder/"episodes.json").read_text())
        assert [e["seed"] for e in episodes]==expected_seeds
        for episode in episodes:
            assert episode["released_rope_force_n"]==0
            if job["group"]!="residual":
                assert episode["residual_rms_rad"]==0
            if episode["success30"]:
                assert episode["hold_sec"]>=30 and episode["final_valid"] and not episode["forbidden"]
                assert episode["max_released_penetration_m"]<=.003 and episode["initial_penetration_m"]<=.001
            raw.append(dict(group=job["group"],training_seed=job["training_seed"],
                bc_pretraining_seed=job.get("bc_pretraining_seed"),**episode))
        groups[job["group"]].append((job["training_seed"],episodes))
    groups={key:sorted(rows,key=lambda x:x[0]) for key,rows in groups.items()}
    assert [r[0] for r in groups["bc"]]==[r[0] for r in groups["bc_ppo"]]
    assert [r[0] for r in groups["bc"]]==[r[0] for r in groups["bc_ppo_last"]]
    assert all(len(groups[g])==(1 if g=="reference" else 3) for g in groups)
    matrices={key:np.asarray([[e["success30"] for e in row] for _,row in rows],float) for key,rows in groups.items()}
    summaries=[]
    for group,matrix in matrices.items():
        rows=[r for r in raw if r["group"]==group]
        summaries.append(dict(group=group,success_rate=float(matrix.mean()),
            successes=int(matrix.sum()),policy_episode_pairs=int(matrix.size),unique_initial_states=HELDOUT_COUNT,
            per_training_seed=matrix.mean(axis=1).tolist(),descriptive_ci95=bootstrap(matrix),
            mean_best_hold_sec=float(np.mean([r["hold_sec"] for r in rows])),
            mean_total_valid_grasp_sec=float(np.mean([r["total_valid_grasp_sec"] for r in rows])),
            mean_streak_break_events=float(np.mean([r["valid_streak_break_events"] for r in rows])),
            max_released_depth_mm=1000*max(r["max_released_penetration_m"] for r in rows),
            max_pre_release_depth_mm=1000*max(r["max_pre_release_penetration_m"] for r in rows),
            peak_force_n=max(r["peak_force_n"] for r in rows),
            successful_peak_force_n=max((r["peak_force_n"] for r in rows if r["success30"]),default=None),
            secondary_all_depth_success_rate=float(np.mean([r["success30_all_depth_le_3mm"] for r in rows])),
            validity_reconstruction_disagreement_frames=sum(r["failure_gates"]["valid_reconstruction_disagreement_frames"] for r in rows),
            failed_by_3s=sum(not r["success30"] and r["duration_sec"]<=3 for r in rows),
            failed_but_alive_40s=sum(not r["success30"] and r["duration_sec"]==40 for r in rows),
            mean_requested_residual_rms_rad=float(np.mean([r["residual_rms_rad"] for r in rows])),
            mean_effective_target_residual_rms_rad=float(np.mean([r["effective_target_residual_rms_rad"] for r in rows])),
            max_effective_target_residual_rad=max(r["max_effective_target_residual_rad"] for r in rows)))
    differences=[]
    paired=[]
    for improved,baseline in (("bc_ppo","bc"),("bc_ppo_last","bc"),("residual","reference")):
        x,y=matrices[improved],matrices[baseline]
        differences.append(dict(contrast=f"{improved} minus {baseline}",difference=float(x.mean()-y.mean()),
            descriptive_ci95=bootstrap(x,y,paired_models=baseline=="bc")))
        for i,(seed,_) in enumerate(groups[improved]):
            paired.append(dict(group=improved,training_seed=seed,
                **transitions(y[i if baseline=="bc" else 0],x[i])))
    modes=[]
    for group in groups:
        for mode in range(6):
            rows=[r for r in raw if r["group"]==group and r["selected_reference"]==mode]
            modes.append(dict(group=group,mode=mode,successes=sum(r["success30"] for r in rows),
                policy_episode_pairs=len(rows),success_rate=float(np.mean([r["success30"] for r in rows])) if rows else None))
    output=args.plan.parent/"report"
    output.mkdir(exist_ok=True)
    failures=[dict(group=group,**row) for group in groups
        for row in failure_gate_summary([r for r in raw if r["group"]==group])]
    write_csv(output/"episodes.csv",[dict(r,failure_gates=json.dumps(r["failure_gates"],sort_keys=True)) for r in raw])
    (output/"episodes.json").write_text(json.dumps(raw,indent=2)+"\n")
    write_csv(output/"failure_gates.csv",failures)
    write_csv(output/"paired_transitions.csv",paired)
    write_csv(output/"orientation_modes.csv",modes)
    result=dict(plan_sha256=digest(args.plan),summaries=summaries,differences=differences,
        paired_transitions=paired,failure_gates=failures,initial_states_exactly_paired=True,
        uncertainty="Descriptive hierarchical bootstrap: resample training origins and shared test cases. Three origins only.",
        audit_scope="All-case outcome/quality statistics; policy/reference/native replay audit covers saved previews only.")
    (output/"statistics.json").write_text(json.dumps(result,indent=2)+"\n")
    plot_results(summaries,differences,output)
    lines=["# S09 PPO增益：独立保留测试", "",
        "任务：固定Orca手抓随机朝向圆柱；初始悬吊3秒后撤除支撑，连续合格抓持至少30秒，40秒结束时仍合格。物体中心只有毫米级扰动，不是任意空间位置抓取。", "",
        f"权重只依据开发集挑选并预先冻结；以下{HELDOUT_COUNT}个初态不参与调参。BC与接续PPO用同一BC起点配对，残差与同一参考回放配对。", "",
        "| 方法 | 成功/执行 | 成功率 | 各训练重复成功率 | 最长合格抓持均值(s) |", "|---|---:|---:|---|---:|"]
    for row in summaries:
        lines.append(f"| {LABELS[row['group']]} | {row['successes']}/{row['policy_episode_pairs']} | {row['success_rate']:.2%} | "+
            " / ".join(f"{v:.2%}" for v in row["per_training_seed"])+f" | {row['mean_best_hold_sec']:.3f} |")
    lines += ["", "![所有重复和末步结果](heldout_success.png)", "", "## PPO到底有没有贡献", ""]
    for row in differences:
        low,high=row["descriptive_ci95"]
        improved,baseline=row["contrast"].split(" minus ")
        verdict="本次平均值提高，描述性区间也高于0，但仍只有三个训练重复。" if low>0 else (
            "本次平均值提高，但区间跨0，尚不能据此认定稳定提高。" if row["difference"]>0 else "本次没有得到平均提升，不能宣称PPO改进成功。")
        lines.append(f"- {LABELS[improved]}相对{LABELS[baseline]}：{100*row['difference']:+.2f}个百分点，描述性95%区间[{100*low:+.2f}, {100*high:+.2f}]。{verdict}")
    lines += ["", "![配对增益与不确定性](paired_gains.png)"]
    lines += ["", "## 接触质量", "",
        "| 方法 | 释放后最大穿透(mm) | 释放前最大穿透(mm) | 全部案例峰值力(N) | 全回合穿透限制下的次级成功率 |",
        "|---|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {LABELS[row['group']]} | {row['max_released_depth_mm']:.3f} | {row['max_pre_release_depth_mm']:.3f} | {row['peak_force_n']:.2f} | {row['secondary_all_depth_success_rate']:.2%} |")
    lines += ["", "## 失败不只有掉落", "",
        "| 方法 | 3秒内终止的失败执行 | 存活满40秒但不达标的执行 | 门控重建不一致帧 |",
        "|---|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {LABELS[row['group']]} | {row['failed_by_3s']} | {row['failed_but_alive_40s']} | {row['validity_reconstruction_disagreement_frames']} |")
    lines += ["", f"failure_gates.csv逐项列出失败执行中的载荷不足、高度损失、穿透等现象及持续时间。原因可同时出现，不能相加当失败总数，也不能仅凭相关计数判定因果。三个模型共用{HELDOUT_COUNT}初态，表中次数是模型与初态的执行对数。浮点边界的重建差异单列，不改写原标签。",
        "", "## 怎样理解", "",
        "- 参考回放组使用同一残差控制接口并固定修正为精确0，避免归一化往返的浮点差异；不是一个训练后的残差策略。",
        "- BC是模仿示范学出完整关节目标；BC+PPO继续通过环境奖励修改这个网络，执行时没有参考动作相加。",
        "- BC+PPO分别列开发集选定的检查点与同三次训练的真正末步；后者检查长训是否退化，不是新增三次独立训练。不能看保留成绩再在两行之间选优，或用中途模型的提高宣称末步也提高。",
        "- 残差执行时为关节目标=参考动作+有界PPO修正；学习修正的实际大小在statistics.json中记录，不能把零修正回放当作残差改进。",
        "- requested是网络请求的修正；effective_target是关节限幅后控制目标相对参考的偏移，不是手指真实转动量。两者都只统计实际执行帧，不能将结束后的补齐帧当动作。完整动作BC策略没有这项加法，其残差字段按约定为0。",
        "- 初始圆柱方向从SO(3)均匀采样；六模式只是选择参考的分类，不是仅测试六个固定角度，也不是所有方向都成功。",
        f"- {3*HELDOUT_COUNT}次模型-初态执行包含三个模型重复使用同{HELDOUT_COUNT}个初态，不是{3*HELDOUT_COUNT}个独立训练。区间用配对初态重采样，训练重复仍较少。",
        "- 数据中training_seed表示PPO训练起点的独立重复编号；对应BC行沿用这个编号方便配对，BC实际预训练种子另列bc_pretraining_seed。逐例seed才是评估初态种子，参考组编号0不表示训练过一次。",
        "- 路径训练量与全部调参搜索花费不同；plan.json记录所选路径，开发日志保留失败。此表不是与纯PPO等总预算比较。",
        "- 本轮没有重新从新RGB视频提取；使用已有处理后DexMV示范及校正参考，隔离训练优化问题。",
        "- 原判据只限制释放后的累计穿透；全回合穿透限制是另报的次级指标。未限制峰值力，因此仿真成功不意味着可直接上硬件。",
        "- 原始逐例数据在episodes.csv及保留嵌套诊断结构的episodes.json，配对新增/丢失在paired_transitions.csv，模式分层在orientation_modes.csv，失败门控在failure_gates.csv。权重与执行哈希见上一级plan.json。", ""]
    (output/"results.md").write_text("\n".join(lines))
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
