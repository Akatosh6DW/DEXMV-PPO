"""Collect auditable evidence for BC+PPO failure hypotheses, not new rankings."""

import csv
import json
from pathlib import Path
import hashlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"


def main():
    inputs = {}

    def read(path):
        raw = path.read_bytes()
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def episodes(run, step=0):
        return read(run/f"step_{step:09d}/episodes.json")

    original = episodes(BASE/"E13_bc_fast_critic")[:32]
    original_cfg = read(BASE/"E13_bc_fast_critic/run_config.json")
    noise = [dict(sampling="deterministic", n=32, successes=sum(r["success30"] for r in original),
                  gained=0, lost=0)]
    for name in ("N01_noise03", "N02_noise01", "N03_noise003"):
        run = BASE/name
        cfg = read(run/"run_config.json")
        probe = read(run/"stochastic_probe.json")
        assert cfg["checkpoint_sha256"] == original_cfg["checkpoint_sha256"] and cfg["steps"] == 0
        after = episodes(run)
        assert [r["seed"] for r in after] == [r["seed"] for r in original]
        with np.load(run/"step_000000000/initial_states.npz") as x, np.load(BASE/"E13_bc_fast_critic/step_000000000/initial_states.npz") as y:
            np.testing.assert_array_equal(x["qpos"], y["qpos"][:32])
        noise.append(dict(sampling=str(probe["sampling_std"]), n=32,
            successes=sum(r["success30"] for r in after),
            gained=sum(b["success30"] and not a["success30"] for a,b in zip(original, after)),
            lost=sum(a["success30"] and not b["success30"] for a,b in zip(original, after))))
    consumed = BASE/"confirmation_retention_20261019/eval"
    bc = episodes(consumed/"bc")
    final = episodes(consumed/"control_last")
    assert [r["seed"] for r in bc] == [r["seed"] for r in final]
    lost = [b for a,b in zip(bc, final) if a["success30"] and not b["success30"] and b["selected_reference"] == 0]
    gates = []
    for gate in lost[0]["failure_gates"]["invalid_seconds_by_gate"]:
        values = [r["failure_gates"]["invalid_seconds_by_gate"][gate] for r in lost]
        gates.append(dict(gate=gate, lost_cases_with_gate=sum(x>0 for x in values),
                          invalid_seconds=sum(values)))
    grad = read(BASE/"D13_mode_gradients/gradient_report.json")
    mini = [r for r in grad["rows"] if isinstance(r["minibatch"], int)]
    pairs = [r["cosine"][0][4] for r in mini if r["cosine"][0][4] is not None]
    gradient_summary = dict(total_minibatches=len(mini), mode0_mode4_present=len(pairs),
        mode0_mode4_negative=sum(x<0 for x in pairs), cosines=pairs,
        mode0_total_sgd_first_order_increase=sum(r["full_batch_sgd_first_order_loss_change"][0]>0 for r in mini),
        limitation="Only early rollout windows at the frozen stage start, raw SGD directions, not historical Adam updates or proof of the cause of later forgetting")
    source_start = episodes(BASE/"E46_bc_budget_continue_s14122")
    source_final = episodes(BASE/"E46_bc_budget_continue_s14122", 655360)
    groups = [("stage_start", source_start), ("single_actor_last", source_final)]
    for name in ("D14_thumb_restore", "D15_non_thumb_restore"):
        run = BASE/name
        metadata = read(run/"joint_restore_config.json")
        assert metadata["optimizer_updates"] == 0
        assert metadata["current_checkpoint_sha256"] == hashlib.sha256((BASE/"E46_bc_budget_continue_s14122/step_000655360/params.pkl").read_bytes()).hexdigest()
        audit = read(run/"step_000000000/native_replay_audit.json")
        assert audit["status"] == "verified" and audit["normalized_action_tolerance"] == 1e-5
        assert audit["teacher_checkpoint_sha256"] == metadata["teacher_checkpoint_sha256"]
        assert audit["joint_restore_config_sha256"] == hashlib.sha256((run/"joint_restore_config.json").read_bytes()).hexdigest()
        assert audit["trace_sha256"] == hashlib.sha256((run/"step_000000000/preview_trajectories.npz").read_bytes()).hexdigest()
        with np.load(run/"step_000000000/initial_states.npz") as x, np.load(BASE/"E46_bc_budget_continue_s14122/step_000655360/initial_states.npz") as y:
            for key in ("seeds", "qpos"):
                np.testing.assert_array_equal(x[key], y[key])
        groups.append((name, episodes(run)))
    joint_rows, mode_rows = [], []
    for name, values in groups:
        assert [r["seed"] for r in values] == [r["seed"] for r in source_final]
        joint_rows.append(dict(group=name, n=len(values), successes=sum(r["success30"] for r in values),
            gained_from_final=sum(b["success30"] and not a["success30"] for a,b in zip(source_final, values)),
            lost_from_final=sum(a["success30"] and not b["success30"] for a,b in zip(source_final, values)),
            max_released_depth_mm=max(r["max_released_penetration_m"] for r in values)*1000,
            max_peak_force_n=max(r["peak_force_n"] for r in values)))
        for mode in range(6):
            rows = [r for r in values if r["selected_reference"]==mode]
            mode_rows.append(dict(group=name, mode=mode, n=len(rows), successes=sum(r["success30"] for r in rows)))
    output = BASE/"report/failure_hypotheses"
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (("noise.csv",noise), ("mode0_lost_gates.csv",gates),
                         ("joint_restore.csv",joint_rows), ("joint_restore_modes.csv",mode_rows)):
        with (output/name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    report = dict(noise=noise, mode0_lost_cases=len(lost), gates=gates,
        gradient=gradient_summary, joint_restore=joint_rows,
        sources=inputs, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="32-case historical frozen-noise probe; consumed 256-case post-hoc failure diagnosis; 64-case development joint intervention. Do not pool cohorts or present joint splices as trained policies.")
    (output/"evidence.json").write_text(json.dumps(report, indent=2)+"\n")
    lines = ["# BC+PPO失败机制：原始证据与诊断干预", "",
        "各节使用不同且明确标记的样本，不能合并算总成功率。关节回退只是两个旧模型的诊断性拼接，没有新的PPO训练，不是最终新方法。", "",
        "## 同权重、零更新的随机动作诊断（32开发初态）", "",
        "| 动作采样 | 成功/32 | 相对确定性新增 | 丢失 |", "|---|---:|---:|---:|"]
    for row in noise:
        lines.append(f"| {row['sampling']} | {row['successes']} | {row['gained']} | {row['lost']} |")
    lines += ["", "## 旧成功抓法具体在哪里丢失（已消耗的256初态）", "",
        f"BC成功而无约束末步失败的mode0案例共{len(lost)}个；以下门控可以重叠。秒数是这些失败回合内的合计，不是因果排序或独立样本量。", "",
        "| 门控 | 出现过的案例数 | 不合格秒数合计 |", "|---|---:|---:|"]
    for row in gates:
        lines.append(f"| {row['gate']} | {row['lost_cases_with_gate']} | {row['invalid_seconds']:.3f} |")
    lines += ["", "## 直接梯度诊断", "",
        f"16个实际大小小批次中，mode0与mode4同时出现{len(pairs)}次，其中方向相反{sum(x<0 for x in pairs)}次。mode0沿总SGD下降方向的一阶损失增加为{gradient_summary['mode0_total_sgd_first_order_increase']}/16。不能把局部负内积直接解释成系统性遗忘的主要原因。",
        "源数据和完整余弦矩阵：../../D13_mode_gradients/gradient_report.json。没有任何策略更新；使用原PPO损失，包括原小批次优势归一化与熵项，各模式梯度求和与原实现按预定浮点容差核对。", "",
        "## 关节回退诊断（64开发初态）", "",
        "| 控制方式 | 成功/64 | 相对末步新增 | 丢失 | 全案例释放后最大穿透mm | 峰值N |", "|---|---:|---:|---:|---:|---:|"]
    for row in joint_rows:
        lines.append(f"| {row['group']} | {row['successes']} | {row['gained_from_final']} | {row['lost_from_final']} | {row['max_released_depth_mm']:.3f} | {row['max_peak_force_n']:.3f} |")
    lines += ["", "thumb_restore只换4个拇指动作；non_thumb_restore换其余13个动作，包含手腕。教师都是E30阶段起点，其他动作仍由E46末步生成，两者每步都读取当前诊断环境状态。所有初态严格配对，保存预览通过权重/原生物理核验。",
        "", "回退成功不能直接证明单关节参数就是根因；其他指头也会改变拇指接触状态，故必须同时看互补回退组。两组都不佳也可能说明整手协调已经改变，而不是证明失载无关。",
        "", "逐朝向结果见joint_restore_modes.csv；输入文件哈希与具体统计见evidence.json。"]
    (output/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
