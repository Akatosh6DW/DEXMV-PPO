"""Record every completed or partial PPO improvement probe, including regressions."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"


def origin_and_budget(run, last_step, base=BASE):
    """Follow selected checkpoints, not discarded training tails, to the origin."""
    current=run.resolve()
    budget=int(last_step)
    seen=set()
    while True:
        if current in seen:
            raise ValueError("Cyclic checkpoint ancestry")
        seen.add(current)
        cfg=json.loads((current/"run_config.json").read_text())
        checkpoint=cfg.get("checkpoint")
        if not checkpoint:
            return current,budget
        checkpoint=Path(checkpoint).resolve()
        parent=checkpoint.parent.parent
        if not parent.is_relative_to(base.resolve()):
            return current,budget
        if not checkpoint.parent.name.startswith("step_"):
            raise ValueError("Continuation needs an explicit checkpoint step")
        budget+=int(checkpoint.parent.name.removeprefix("step_"))
        current=parent


def main():
    output = BASE/"report"
    output.mkdir(exist_ok=True)
    summaries, curves = [], []
    for run in sorted(BASE.glob("E*")):
        if not (run/"metrics.csv").exists():
            continue
        with (run/"metrics.csv").open() as f:
            rows = [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
        cfg = json.loads((run/"run_config.json").read_text())
        meta = json.loads((run/"improvement_config.json").read_text()) if (run/"improvement_config.json").exists() else {}
        constraint = json.loads((run/"constraint_config.json").read_text()) if (run/"constraint_config.json").exists() else {}
        warmup = json.loads((run/"warmup_config.json").read_text()) if (run/"warmup_config.json").exists() else {}
        feedback = json.loads((run/"feedback_config.json").read_text()) if (run/"feedback_config.json").exists() else {}
        rehearsal = json.loads((run/"rehearsal_config.json").read_text()) if (run/"rehearsal_config.json").exists() else {}
        temporal = json.loads((run/"temporal_config.json").read_text()) if (run/"temporal_config.json").exists() else {}
        adapter = json.loads((run/"adapter_config.json").read_text()) if (run/"adapter_config.json").exists() else {}
        exploration = json.loads((run/"exploration_config.json").read_text()) if (run/"exploration_config.json").exists() else {}
        optimizer = [json.loads(line) for line in (run/"ppo_metrics.jsonl").read_text().splitlines()] if (run/"ppo_metrics.jsonl").exists() else []
        last_optimizer = next((r for r in reversed(optimizer) if "training/kl_mean" in r),{})
        first_episodes = json.loads((run/f"step_{int(rows[0]['step']):09d}"/"episodes.json").read_text())
        last_episodes = json.loads((run/f"step_{int(rows[-1]['step']):09d}"/"episodes.json").read_text())
        if [e["seed"] for e in first_episodes] != [e["seed"] for e in last_episodes]:
            raise ValueError(f"Unpaired evaluation seeds in {run}")
        won = sum(not a["success30"] and b["success30"] for a,b in zip(first_episodes,last_episodes))
        lost = sum(a["success30"] and not b["success30"] for a,b in zip(first_episodes,last_episodes))
        successes = [e for e in last_episodes if e["success30"]]
        best = max(rows, key=lambda r:r["success30"])
        origin,budget=origin_and_budget(run,rows[-1]["step"])
        origin_config=json.loads((origin/"run_config.json").read_text())
        origin_checkpoint=origin_config.get("checkpoint")
        initialization_path=Path(origin_checkpoint).parent.parent/"critic_pretraining.json" if origin_checkpoint else None
        initialization=json.loads(initialization_path.read_text()) if initialization_path and initialization_path.exists() else {}
        collection_steps=int(initialization.get("actual_environment_interactions",0))
        baseline=json.loads((origin/"step_000000000/episodes.json").read_text())
        if [e["seed"] for e in baseline] != [e["seed"] for e in last_episodes]:
            raise ValueError(f"Unpaired original baseline in {run}")
        baseline_rate=sum(e["success30"] for e in baseline)/len(baseline)
        summaries.append(dict(run=run.name, method=cfg["method"], complete=(run/"summary.json").exists(),
            interrupted=(run/"interruption.json").exists(),
            logged_interactions_lower_bound=max(int(rows[-1]["step"]),max((int(r["step"]) for r in optimizer),default=0)),
            seed=cfg["seed"], eval_start=cfg["eval_start"], eval_count=cfg["eval_count"],
            first=rows[0]["success30"], last=rows[-1]["success30"], max=best["success30"], best_step=int(best["step"]),
            last_step=int(rows[-1]["step"]), delta=rows[-1]["success30"]-rows[0]["success30"],
            hold_first=rows[0]["hold_sec"], hold_last=rows[-1]["hold_sec"],
            hold_max=max(r["hold_sec"] for r in rows),
            hold_best_step=int(max(rows,key=lambda r:r["hold_sec"])["step"]),
            final_depth_mm=1000*rows[-1]["max_penetration_m"], final_peak_force_n=rows[-1]["peak_force_n"],
            discount=meta.get("training_overrides",{}).get("discounting"),
            actor_learning_rate=cfg["learning_rate"],
            critic_learning_rate=constraint.get("critic_learning_rate") or cfg["learning_rate"],
            initial_std_override=meta.get("initial_std_override"), configured_std=cfg["std"],
            teacher_kl_weight=constraint.get("teacher_kl_weight",0),
            transition_bookkeeping_fixed=(run/"transition_config.json").exists(),
            actor_warmup_transitions=warmup.get("actor_warmup_transitions",0),
            new_input_update_multiplier=feedback.get("new_input_update_multiplier",1.),
            demo_rehearsal_weight=rehearsal.get("demo_weight",0),
            demo_rehearsal_dataset_sha256=rehearsal.get("dataset_sha256"),
            unroll_length=temporal.get("unroll_length",20),
            sequence_batch_size=temporal.get("batch_size",32),
            actor_update_rule=adapter.get("actor_update_rule","all_actor_parameters"),
            acquisition_noise_scale=exploration.get("acquisition_noise_scale",1.),
            last_update_kl=last_optimizer.get("training/kl_mean"),
            last_value_loss=last_optimizer.get("training/v_loss"),
            gained_cases=won, lost_cases=lost,
            original_baseline_run=origin.name, original_baseline_success=baseline_rate,
            delta_from_original_baseline=rows[-1]["success30"]-baseline_rate,
            gained_from_original=sum(not a["success30"] and b["success30"] for a,b in zip(baseline,last_episodes)),
            lost_from_original=sum(a["success30"] and not b["success30"] for a,b in zip(baseline,last_episodes)),
            selected_model_path_online_steps=budget,
            additional_critic_collection_steps=collection_steps,
            selected_model_path_total_environment_interactions=budget+collection_steps,
            original_bc_checkpoint=initialization.get("source_checkpoint",origin_checkpoint),
            final_residual_rms_rad=rows[-1]["residual_rms_rad"],
            successful_max_depth_mm=max((1000*e["max_released_penetration_m"] for e in successes), default=None),
            successful_peak_force_n=max((e["peak_force_n"] for e in successes), default=None),
            reward_version=meta.get("reward_version",cfg["env_config"]["reward_version"]), checkpoint=cfg.get("checkpoint")))
        curves.extend(dict(run=run.name, **r) for r in rows)
    for name, rows in (("iterations",summaries),("curves",curves)):
        (output/f"{name}.json").write_text(json.dumps(rows,indent=2)+"\n")
        if rows:
            with (output/f"{name}.csv").open("w") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines=["# S09 PPO增益开发监测", "", "仅为开发数据，不是冻结后的最终保留测试；partial不作为最终结论。", "",
        "| 运行 | 状态 | 初始成功率 | 末步成功率 | 最大成功率 | 最佳步数 | 末步穿透mm |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in summaries:
        status="complete" if r["complete"] else "interrupted" if r["interrupted"] else "partial"
        lines.append(f"| {r['run']} | {status} | {r['first']:.2%} | "
                     f"{r['last']:.2%} | {r['max']:.2%} | {r['best_step']} | {r['final_depth_mm']:.3f} |")
    lines += ["", "初始值是本次运行的起点。恢复检查点可能是BC，也可能已接过PPO；接续增益不能直接当作相对原BC的总增益。",
              "接续的累计交互预算须加上前序训练；例如E24还要加上E19的163840步。残差初始值须结合修正RMS确认是否为零修正参考。",
              "要证明改进，需比较配对初态上的最终模型与起点，不能只挑曲线最高值。"]
    lines += ["", f"本表E系列已记录PPO交互总量下界：{sum(r['logged_interactions_lower_bound'] for r in summaries):,}。",
        "包含中断后重跑的作废尾段和各接续阶段，不是某个模型路径的训练量；不含示范收集、BC/critic监督更新、smoke、评估或未写日志的交互。interrupted表示进程已核对不存在，partial仅表示尚无完成标记，不单靠文件推断仍在运行。"]
    (output/"development.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(report=str(output), runs=[{k:r[k] for k in
        ("run","complete","first","last","max","gained_cases","lost_cases")} for r in summaries]),indent=2))


if __name__ == "__main__":
    main()
