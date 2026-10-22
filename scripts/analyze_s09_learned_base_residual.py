"""Report additional residual learning separately from inherited base gains."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_s09_bc_confirmation import write_csv
from analyze_s09_late_experts import transitions
from analyze_s09_angular_feedback import paired_initials
from retarget_s09_reference import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    read = lambda path:json.loads(path.read_text())
    cfg = read(args.run/"run_config.json")
    protocol = read(args.run/"learned_base_config.json")
    experts_path = args.run/"residual_expert_config.json"
    residual_actors = read(experts_path)["residual_expert_count"] if experts_path.exists() else 1
    summary = read(args.run/"summary.json")
    assert summary["status"] == "completed" and cfg["method"] == "residual"
    assert digest(Path(protocol["base_checkpoint"])) == protocol["base_sha256"]
    if cfg["checkpoint"]:
        raise ValueError("Continuation needs explicit ancestor residual-budget accounting")
    selection_path = Path(protocol["base_checkpoint"]).parent.parent/"selection.json"
    selection = read(selection_path)
    assert selection["candidate_sha256"] == protocol["base_sha256"]
    inherited = selection["inherited_path_online_steps"]+selection["extra_ppo_steps"]
    raw,rows,modes,evidence = [],[],[],{}
    by_step = {}
    initial = None
    for metric in summary["rows"]:
        step = metric["step"]
        folder = args.run/f"step_{step:09d}"
        with np.load(folder/"initial_states.npz",allow_pickle=False) as z:
            if initial is None:
                initial = dict(z)
            paired_initials(initial,z)
        eps = read(folder/"episodes.json")
        assert len(eps) == cfg["eval_count"]
        assert np.mean([r["success30"] for r in eps]) == metric["success30"]
        audit = read(folder/"native_replay_audit.json")
        assert audit["status"] == "verified" and audit["normalized_action_tolerance"] == 1e-5
        assert audit["base_sha256"] == protocol["base_sha256"]
        assert audit["checkpoint_sha256"] == digest(folder/"params.pkl")
        assert audit["trace_sha256"] == digest(folder/"preview_trajectories.npz")
        if step == 0 and not cfg["checkpoint"]:
            assert metric["residual_rms_rad"] == 0
        by_step[step] = eps
        rows.append(dict(step=step,success30=metric["success30"],success_count=sum(r["success30"] for r in eps),
            hold_sec=metric["hold_sec"],max_penetration_mm=1000*metric["max_penetration_m"],
            peak_force_n=metric["peak_force_n"],residual_rms_rad=metric["residual_rms_rad"],
            forbidden_rate=metric["forbidden_rate"],released_rope_force_n=metric["released_rope_force_n"]))
        raw.extend(dict(step=step,**r) for r in eps)
        for mode in range(6):
            part = [r for r in eps if r["selected_reference"] == mode]
            modes.append(dict(step=step,mode=mode,n=len(part),successes=sum(r["success30"] for r in part)))
        evidence[str(step)] = dict(episodes_sha256=digest(folder/"episodes.json"),audit_sha256=digest(folder/"native_replay_audit.json"))
    pairs = [dict(step=step,**transitions(by_step[0],eps)) for step,eps in by_step.items()]
    result = dict(status="completed_development_only",success30=summary["success30"],hold_sec=summary["hold_sec"],
        paired=pairs,rows=rows,evidence=evidence,base_sha256=protocol["base_sha256"],residual_actors=residual_actors,
        inherited_online_steps=inherited,additional_online_steps=summary["final_step"],
        total_online_steps=inherited+summary["final_step"],selection_sha256=digest(selection_path),
        limitations="One PPO seed, development states. Learned BC+PPO portfolio base, not pure BC or timed-video reference. Zero-residual old-runtime bitwise identity failed; use this interface's actual start.",
        source_sha256=digest(Path(__file__)))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"statistics.json").write_text(json.dumps(result,indent=2)+"\n")
    for name,values in (("curve.csv",rows),("paired.csv",pairs),("orientation.csv",modes),("episodes.csv",raw)):
        write_csv(args.output/name,values)
    lines = ["# 学习基础策略＋残差PPO：开发结果", "",
        f"冻结已BC/PPO训练的历史组合，新增{residual_actors}个残差动作网络；物理、奖励与严格30秒成功条件不变。", "",
        "| 追加步 | 成功数 | 成功率 | 平均合格秒 | 最大穿透mm | 峰值N | 修正RMS rad |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['step']} | {row['success_count']}/{cfg['eval_count']} | {row['success30']:.2%} | {row['hold_sec']:.3f} | {row['max_penetration_mm']:.3f} | {row['peak_force_n']:.3f} | {row['residual_rms_rad']:.6f} |")
    lines += ["",f"first/last/max/best_step：{summary['success30']}","", "| 追加步 | 相对零步新增 | 相对零步丢失 |", "|---|---:|---:|"]
    for row in pairs:
        lines.append(f"| {row['step']} | {row['gained']} | {row['lost']} |")
    lines += ["",f"继承{inherited}在线步，新增{summary['final_step']}步；不把基础已有提升算成残差学习收益。",
        "保存预览通过权重/动作/历史语义/物理审计，未保存的评估轨迹不声称逐帧回放核验。最大穿透和峰值包含失败案例。",
        "旧运行时与零残差运行时不是逐位同轨迹，所以本表以新接口自身实测零步为基线，不直接借用旧成功率。",
        "单训练种子、重复开发集；不是新的未见测试或多训练种子稳定性证据。"]
    (args.output/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
