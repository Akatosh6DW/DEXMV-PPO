"""Summarize post-confirmation development probes against matched controls.

This reads only development evaluations. It never selects a new checkpoint
using the already consumed confirmation set or overwrites that report.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"
CONTROLS = {
    "E57_residual_acquisition_exploration": ("E43_residual_temporal80", "起手探索倍率10"),
    "E58_bc_persistence_fast_critic": ("E13_bc_fast_critic", "连续抓持训练奖励R2"),
    "E59_bc_reliable_mode_anchor": ("E13_bc_fast_critic", "仅mode0的BC约束0.1"),
    "E60_bc_global_anchor_control": ("E13_bc_fast_critic", "全朝向BC约束0.1"),
    "E61_residual_long_credit": ("E43_residual_temporal80", "GAE lambda=1"),
    "E62_bc_long_credit": ("E42_bc_temporal80", "GAE lambda=1"),
    "E63_bc_training_jitter5": ("E13_bc_fast_critic", "仅训练位置扰动扩大到5mm"),
    "E64_residual_training_jitter5": ("E43_residual_temporal80", "仅训练位置扰动扩大到5mm"),
    "E65_bc_training_jitter5_long": ("E19_bc_fast_critic_long", "训练5mm扰动长预算验证"),
    "E66_bc_training_jitter5_s14102": ("E27_bc_fast_critic_s14102", "训练5mm扰动第二训练种子"),
    "E67_bc_pose_experts": ("E68_bc_pose_expert_control", "独立架构：六份朝向actor，参数量6倍"),
    "E69_bc_update_reuse1": ("E68_bc_pose_expert_control", "每批数据复用4遍改1遍"),
    "E70_bc_critic_history": ("E58_bc_persistence_fast_critic", "R2只给critic补充奖励历史量"),
    "E71_bc_demo_weight1000": ("E41_bc_demo_rehearsal", "示范复习权重10改1000"),
    "E72_bc_stage_retention": ("E46_bc_budget_continue_s14122", "长训阶段只保护起点熟练朝向0/3/5"),
    "E73_bc_discount999_split": ("E13_bc_fast_critic", "分离学习率下gamma0.99改0.999"),
    "E74_bc_record_hold": ("E68_bc_pose_expert_control", "主奖励仅计最长连续合格时长增长"),
    "E75_bc_head_only": ("E68_bc_pose_expert_control", "冻结隐藏层，仅训练均值/尺度输出层"),
}


def metrics(run):
    with (run/"metrics.csv").open() as stream:
        return [{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]


def bc_sampling_result():
    run = BASE/"B01_bc_mode_balanced_s13102"
    if not (run/"summary.json").exists():
        return None
    original = BASE.parent/"I27_phase_s13102"
    cfg, old = [json.loads((p/"run_config.json").read_text()) for p in (run, original)]
    for key in ("seed", "std", "bc_updates", "dataset_sha256", "model_sha256", "reference_sha256", "env_config"):
        assert cfg[key] == old[key], key
    assert cfg["steps"] == 0 and (cfg["eval_start"], cfg["eval_count"]) == (130000, 64)
    control = BASE/"E68_bc_pose_expert_control/step_000000000"
    old_ppo_cfg = json.loads((control.parent/"run_config.json").read_text())
    assert old_ppo_cfg["checkpoint_sha256"] == hashlib.sha256((original/"step_000000000/params.pkl").read_bytes()).hexdigest()
    with np.load(run/"step_000000000/initial_states.npz") as current, np.load(control/"initial_states.npz") as previous:
        for key in ("qpos", "seeds"):
            np.testing.assert_array_equal(current[key], previous[key])
    before = json.loads((control/"episodes.json").read_text())
    after = json.loads((run/"step_000000000/episodes.json").read_text())
    by_seed = {r["seed"]:r for r in after}
    record = metrics(run)[0]
    assert sum(r["success30"] for r in after)/64 == record["success30"]
    result = dict(run=run.name, bc_seed=cfg["seed"], bc_updates=cfg["bc_updates"],
        old_bc_success=sum(r["success30"] for r in before)/64, balanced_bc_success=record["success30"],
        gained=sum(by_seed[r["seed"]]["success30"] and not r["success30"] for r in before),
        lost=sum(not by_seed[r["seed"]]["success30"] and r["success30"] for r in before),
        max_all_case_released_depth_mm=record["max_penetration_m"]*1000,
        max_all_case_peak_force_n=record["peak_force_n"])
    continuation = BASE/"E76_bc_balanced_origin"
    result["ppo_status"] = "partial" if continuation.exists() else "not_started"
    if (continuation/"summary.json").exists():
        current_cfg = json.loads((continuation/"run_config.json").read_text())
        for key in ("seed", "std", "learning_rate", "steps", "evals", "eval_count", "eval_start", "env_config"):
            assert current_cfg[key] == old_ppo_cfg[key], key
        assert current_cfg["checkpoint_sha256"] == hashlib.sha256((run/"step_000000000/params.pkl").read_bytes()).hexdigest()
        for path in (continuation, control.parent):
            assert json.loads((path/"constraint_config.json").read_text())["critic_learning_rate"] == .0003
        values = metrics(continuation)
        assert values[0]["success30"] == record["success30"]
        assert values[-1]["step"] == 40960
        ppo_episodes = json.loads((continuation/"step_000040960/episodes.json").read_text())
        ppo = {row["seed"]:row for row in ppo_episodes}
        result.update(ppo_status="complete", ppo_last=values[-1]["success30"],
            original_pipeline_ppo_last=metrics(control.parent)[-1]["success30"],
            ppo_gained_from_own_bc=sum(ppo[row["seed"]]["success30"] and not row["success30"] for row in after),
            ppo_lost_from_own_bc=sum(not ppo[row["seed"]]["success30"] and row["success30"] for row in after),
            ppo_max_all_case_released_depth_mm=values[-1]["max_penetration_m"]*1000,
            ppo_max_all_case_peak_force_n=values[-1]["peak_force_n"])
    return result


def main():
    rows = []
    for name, (control_name, change) in CONTROLS.items():
        run, control = BASE/name, BASE/control_name
        if not (run/"metrics.csv").exists():
            continue
        cfg, old_cfg = [json.loads((p/"run_config.json").read_text()) for p in (run, control)]
        for key in ("method", "seed", "std", "learning_rate", "steps", "eval_count", "eval_start",
                    "model_sha256", "reference_sha256", "checkpoint_sha256", "env_config"):
            # Reward shaping explicitly changes training reward metadata only.
            if key == "checkpoint_sha256" and name.startswith("E67_"):
                from hashlib import sha256
                derived = Path(cfg["checkpoint"])
                conversion = json.loads((derived.parent/"conversion.json").read_text())
                assert conversion["source_checkpoint_sha256"] == old_cfg[key]
                assert conversion["derived_checkpoint_sha256"] == cfg[key]
                assert sha256(derived.read_bytes()).hexdigest() == cfg[key]
                assert conversion["all_six_actor_copies_bitwise_identical"]
                assert conversion["critic_and_normalizer_unchanged"]
            elif key == "env_config" and name.startswith("E74_"):
                current, old = dict(cfg[key]), dict(old_cfg[key])
                assert current["reward_version"] == "S09-R6_record_hold"
                current["reward_version"] = old["reward_version"]
                assert current == old, (name, key)
            elif key == "env_config" and name.startswith("E70_"):
                current = dict(cfg[key])
                assert current.pop("critic_history_features") == "clean_77_hold_progress_94"
                assert current == old_cfg[key], (name, key)
            elif key == "env_config" and name.startswith("E58_"):
                current, old = dict(cfg[key]), dict(old_cfg[key])
                current.pop("hold_progress_weight", None)
                current["reward_version"] = old["reward_version"]
                assert current == old, (name, key)
            else:
                assert cfg.get(key) == old_cfg.get(key), (name, key)
        values, baseline = metrics(run), metrics(control)
        optimizer, old_optimizer = [json.loads((p/"constraint_config.json").read_text())
                                    for p in (run, control)]
        assert optimizer["critic_learning_rate"] == old_optimizer["critic_learning_rate"]
        with np.load(run/"step_000000000/initial_states.npz") as initial, \
                np.load(control/"step_000000000/initial_states.npz") as old_initial:
            for key in ("seeds", "qpos"):
                np.testing.assert_array_equal(initial[key], old_initial[key])
        assert values[0]["success30"] == baseline[0]["success30"]
        complete = (run/"summary.json").exists()
        if complete:
            assert values[-1]["step"] == cfg["steps"] == baseline[-1]["step"]
        best = max(values, key=lambda r:r["success30"])
        rows.append(dict(run=name, method=cfg["method"], change=change,
            planned_steps=cfg["steps"],
            matched_control=control_name, status="complete" if complete else "partial",
            first=values[0]["success30"], last=values[-1]["success30"],
            max=best["success30"], best_step=int(best["step"]), last_step=int(values[-1]["step"]),
            matched_control_last=baseline[-1]["success30"],
            gain_from_own_start_pp=100*(values[-1]["success30"]-values[0]["success30"]) if complete else None,
            gain_from_matched_control_pp=100*(values[-1]["success30"]-baseline[-1]["success30"]) if complete else None,
            final_all_case_depth_mm=1000*values[-1]["max_penetration_m"] if complete else None,
            final_all_case_peak_force_n=values[-1]["peak_force_n"] if complete else None))
    output = BASE/"report"
    output.mkdir(exist_ok=True)
    if rows:
        with (output/"post_confirmation_probes.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# 独立确认后的开发探测", "",
        "以下仅使用130000--130063开发初态，不是新的独立验证。每项与对应控制计划在线训练预算相同；partial不得与控制末步比较。E67增加actor容量六倍，属于独立架构组；E69减少梯度更新数，并非等梯度计算量。", "",
        "| 试验 | 改动 | 计划步数 | 状态 | 起点 | 末步 | 匹配控制末步 | 相对控制(pp) |",
        "|---|---|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        last = f"{row['last']:.2%}" if row["status"] == "complete" else "未完成"
        delta = f"{row['gain_from_matched_control_pp']:+.2f}" if row["status"] == "complete" else "不比较"
        lines.append(f"| {row['run']} | {row['change']} | {row['planned_steps']} | {row['status']} | {row['first']:.2%} | {last} | {row['matched_control_last']:.2%} | {delta} |")
    lines += ["", "pp表示百分点，不是相对百分比。与控制持平或下降不支持该改动有效；高于控制但低于自己的BC/参考起点也不算解决目标。",
        "", "原始数据位于此report文件夹的上一级目录内，对应各E目录。CSV额外列出first/last/max/best_step及全部案例的末步穿透、峰值力；失败异常不删除。",
        "", "全局teacher_kl字段不能代替E59的mode_anchor_config；GAE参数见credit_config/temporal_config；训练与评估扰动分别见training_jitter_config。E70新增critic历史量及对应中性初始化见critic_history_config；E71强示范权重见rehearsal_config。每项的实际配置和源码快照保存在对应运行目录。",
        "", "[此前已冻结的独立确认](../confirmation_20261019/report/results.md)与本开发表分开，不能将此处新候选替换进旧确认表。"]
    bc = bc_sampling_result()
    if bc is not None:
        with (output/"bc_origin_sampling.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(bc))
            writer.writeheader()
            writer.writerow(bc)
        lines += ["", "## 独立BC采样候选", "",
            "该行没有在线PPO训练，不能把BC起点变化算作PPO收益；验证集不进入训练，没有新增视频、示范或缺失朝向样本。", "",
            "| 运行 | BC种子 | BC更新数 | 原BC | 类别均衡BC | 新增/丢失 |",
            "|---|---:|---:|---:|---:|---:|",
            f"| {bc['run']} | {bc['bc_seed']} | {bc['bc_updates']} | {bc['old_bc_success']:.2%} | {bc['balanced_bc_success']:.2%} | {bc['gained']}/{bc['lost']} |",
            "", f"全部案例释放后最大穿透{bc['max_all_case_released_depth_mm']:.3f}mm、峰值力{bc['max_all_case_peak_force_n']:.3f}N。采样复制只改变已有训练帧的抽样权重，不增加独立样本数量。"]
        if bc["ppo_status"] == "complete":
            lines += ["", f"E76接续40960步PPO后为{bc['ppo_last']:.2%}，相对自己的均衡BC新增{bc['ppo_gained_from_own_bc']}、丢失{bc['ppo_lost_from_own_bc']}；原BC起点同配方PPO控制为{bc['original_pipeline_ppo_last']:.2%}。前一比较才是PPO自身净收益，后一比较是改变BC采样后整条管线的差别。",
                      f"PPO末步全部案例释放后最大穿透{bc['ppo_max_all_case_released_depth_mm']:.3f}mm、峰值力{bc['ppo_max_all_case_peak_force_n']:.3f}N。仍只是一条开发训练，不是新保留测试。"]
        else:
            lines += ["", f"接续PPO状态：{bc['ppo_status']}，尚不填入末步成绩。"]
    (output/"post_confirmation_probes.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(rows=len(rows), report=str(output/"post_confirmation_probes.md"))))


if __name__ == "__main__":
    main()
