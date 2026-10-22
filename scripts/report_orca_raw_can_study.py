"""Audit and summarize all new-video jobs, including failures and matched seeds."""

import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"Aero-Hand/sim_rl/mujoco_playground"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.stats import beta

from run_orca_suite_job import summarize

OUT = ROOT/"artifacts/orca_cylinder_raw_video_20261009"
LABELS = dict(ppo="PPO", guided="BC + PPO", residual="Reference + residual PPO",
              bc="BC only", zero="Reference only")
ZH = dict(ppo="纯 PPO", guided="BC 后 PPO", residual="参考加残差 PPO", bc="仅 BC", zero="仅参考")
COLORS = dict(ppo="#3268a8", guided="#208561", residual="#ba593d", bc="#777777", zero="#87699f")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(k, n):
    return (0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            1. if k == n else float(beta.ppf(.975, k+1, n-k)))


def main():
    protocol = json.loads((OUT/"protocol.json").read_text())
    for path, expected in protocol["frozen_sha256"].items():
        if digest(ROOT/path) != expected:
            raise ValueError("Frozen input changed: " + path)
    for phase in ("train", "eval"):
        complete = json.loads((OUT/(phase+"_completion.json")).read_text())
        assert complete["complete"] and complete["protocol_sha256"] == digest(OUT/"protocol.json")
        assert len(complete["records"]) == len(protocol[phase+"_jobs"])
    report = OUT/"report"
    report.mkdir(exist_ok=True)
    runs, episodes, curves, bc_losses = [], [], [], []
    paired, preview_count = {}, 0
    for phase in ("train", "eval"):
        for path in protocol[phase+"_jobs"]:
            job = json.loads((ROOT/path).read_text())
            run = ROOT/job["output"]
            config = json.loads((run/"run_config.json").read_text())
            result = json.loads((run/"summary.json").read_text())
            assert result["status"] == "completed" and result["actual_steps"] == job["steps"]
            assert result["job_sha256"] == config["job_sha256"] == digest(ROOT/path)
            for key, value in job.items():
                assert config[key] == value, (path, key)
            for source, expected in config["source_sha256"].items():
                assert digest(run/"sources"/source) == expected
                assert digest(ROOT/"scripts"/source) == expected
            assert digest(run/"model.xml") == config["model_sha256"] == digest(ROOT/job["model"])
            assert digest(run/"reference.npz") == config["reference_sha256"] == digest(ROOT/job["reference"])
            if phase == "train":
                assert not job.get("checkpoint") and "fresh PRNG" in config["initialization"]
                if job["method"] == "guided":
                    bc_losses.extend(dict(train_seed=job["seed"], **row)
                                     for row in json.loads((run/"bc_loss.json").read_text()))
            else:
                if job.get("checkpoint"):
                    assert (ROOT/job["checkpoint"]).resolve().is_relative_to(OUT)
                    assert digest(ROOT/job["checkpoint"]) == config["checkpoint_sha256"]
            model = mujoco.MjModel.from_xml_path(str(run/"model.xml"))
            assert int(model.geom("cube").type[0]) == mujoco.mjtGeom.mjGEOM_CYLINDER
            np.testing.assert_allclose(model.geom("cube").size[:2], [.069644/2, .103637/2])
            env = SimpleNamespace(arm=config["arm"], residual_bound=.08,
                                  _lowers=model.actuator_ctrlrange[:, 0], _uppers=model.actuator_ctrlrange[:, 1])
            with (run/"metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            expected_steps = [0, 163840, 327680, 491520, 655360] if phase == "train" else [0]
            assert [int(r["step"]) for r in rows] == expected_steps
            seeds = list(range(job["eval_start"], job["eval_start"]+job["eval_count"]))
            for row in rows:
                folder = run/f"step_{int(row['step']):09d}"
                data = json.loads((folder/"episodes.json").read_text())
                assert [e["seed"] for e in data] == seeds
                assert np.isclose(float(row["clean_success30"]), np.mean([e["clean_success30"] for e in data]))
                for ep in data:
                    assert ep["clean_success30"] == (ep["success30"] and not ep["substep_forbidden"]
                                                     and ep["post_release_penetration_m"] <= .003)
                with np.load(folder/"initial_states.npz", allow_pickle=False) as z:
                    np.testing.assert_array_equal(z["seeds"], seeds)
                    key = (phase, job["benchmark"], tuple(seeds))
                    if key in paired:
                        np.testing.assert_array_equal(paired[key], z["qpos"])
                    else:
                        paired[key] = z["qpos"].copy()
                with np.load(folder/"preview_trajectories.npz", allow_pickle=False) as z:
                    trace = dict(z)
                assert all(np.isfinite(value).all() for value in trace.values())
                _, checked = summarize(trace, env, seeds[:4])
                for a, b in zip(checked, data[:4]):
                    for name in ("success30", "clean_success30", "final_valid", "substep_forbidden"):
                        assert a[name] == b[name]
                    for name in ("longest_hold_sec", "post_release_penetration_m", "residual_rms_rad"):
                        assert np.isclose(a[name], b[name], atol=1e-6)
                preview_count += len(checked)
                if env.arm == "pure":
                    np.testing.assert_array_equal(trace["video_base"], 0.)
                if job["method"] == "zero":
                    np.testing.assert_allclose(trace["ctrl"], trace["video_base"], atol=1e-5)
                if job["benchmark"] == "B5_push":
                    np.testing.assert_allclose(np.linalg.norm(trace["push"][:, 200:204], axis=-1), .15, atol=1e-6)
                if phase == "train":
                    curves.append(dict(method=job["method"], train_seed=job["seed"], step=int(row["step"]),
                                       clean_success30=float(row["clean_success30"]), mean_hold=float(row["mean_hold"]),
                                       mean_return=float(row["mean_return"]), elapsed_sec=float(row["elapsed_sec"])))
            if phase == "eval":
                common = dict(method=job["method"], train_seed=job["seed"], budget=job["budget"], condition=job["benchmark"])
                episodes.extend(dict(common, **ep) for ep in data)
                successes = sum(ep["clean_success30"] for ep in data)
                lo, hi = interval(successes, len(data))
                runs.append(dict(common, successes=successes, episodes=len(data), clean_success30=successes/len(data),
                                 conditional_ci95_low=lo, conditional_ci95_high=hi,
                                 raw_success30=float(np.mean([e["success30"] for e in data])),
                                 success15=float(np.mean([e["success15"] for e in data])),
                                 mean_hold=float(np.mean([e["longest_hold_sec"] for e in data])),
                                 max_hold=max(e["longest_hold_sec"] for e in data),
                                 drop_rate=float(np.mean([e["drop"] for e in data])),
                                 valid_before_push_rate=float(np.mean([e["valid_before_push"] for e in data])),
                                 valid_after_push_rate=float(np.mean([e["valid_one_second_after_push"] for e in data])),
                                 forbidden_rate=float(np.mean([e["substep_forbidden"] for e in data])),
                                 depth_failure_rate=float(np.mean([e["post_release_penetration_m"] > .003 for e in data])),
                                 mean_peak_force_n=float(np.mean([e["substep_force_peak_n"] for e in data])),
                                 mean_residual_rms_rad=float(np.mean([e["residual_rms_rad"] for e in data]))))
    assert (len(runs), len(episodes), len(curves), preview_count) == (66, 4224, 45, 444)
    groups = []
    for condition, method, budget in sorted({(r["condition"], r["method"], r["budget"]) for r in runs}):
        items = [r for r in runs if (r["condition"], r["method"], r["budget"]) == (condition, method, budget)]
        rates = [r["clean_success30"] for r in items]
        groups.append(dict(condition=condition, method=method, budget=budget,
                           trained_seeds=0 if method == "zero" else len(items),
                           episodes=sum(r["episodes"] for r in items), successes=sum(r["successes"] for r in items),
                           mean_success=float(np.mean(rates)), seed_std=float(np.std(rates, ddof=1)) if len(rates)>1 else None,
                           mean_raw_success30=float(np.mean([r["raw_success30"] for r in items])),
                           mean_raw_success15=float(np.mean([r["success15"] for r in items])),
                           mean_hold=float(np.mean([r["mean_hold"] for r in items])),
                           drop_rate=float(np.mean([r["drop_rate"] for r in items])),
                           forbidden_rate=float(np.mean([r["forbidden_rate"] for r in items])),
                           depth_failure_rate=float(np.mean([r["depth_failure_rate"] for r in items]))))
    differences = []
    for condition in ("B0_nominal", "B1_position", "B5_push"):
        for budget in (163840, 655360):
            for method, comparator, comp_budget in (("guided", "ppo", budget), ("residual", "ppo", budget),
                                                   ("guided", "bc", 0), ("residual", "zero", 0)):
                values = []
                for seed in (51001, 51002, 51003):
                    def rate(m, b, s):
                        return next(r["clean_success30"] for r in runs if
                                    (r["condition"], r["method"], r["budget"], r["train_seed"]) == (condition, m, b, s))
                    values.append(rate(method, budget, seed)-rate(comparator, comp_budget, 51000 if comparator == "zero" else seed))
                d = np.asarray(values)
                perm = [abs(np.mean(d*signs)) for signs in itertools.product((-1, 1), repeat=3)]
                differences.append(dict(condition=condition, budget=budget, method=method, comparator=comparator,
                                        differences=values, mean_difference=float(d.mean()),
                                        descriptive_signflip_p=float(np.mean(np.asarray(perm) >= abs(d.mean())-1e-12))))
    for name, data in (("runs", runs), ("episodes", episodes), ("groups", groups), ("learning_curves", curves),
                       ("bc_losses", bc_losses)):
        save_csv(report/(name+".csv"), data)
    (report/"paired_differences.json").write_text(json.dumps(differences, indent=2)+"\n")
    audit = dict(complete=True, train_runs=9, eval_jobs=66, test_episodes=4224,
                 preview_episodes_rescored=preview_count, protocol_sha256=digest(OUT/"protocol.json"),
                 scope="All configurations and episode scores checked; only four saved full traces per job/step independently rescored")
    (OUT/"audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    with np.load(OUT/"perception/tomato41/hand_landmarks.npz") as hand:
        world = hand["world_landmarks"]
    with np.load(OUT/"perception/can41_v2/hand_object.npz") as packet:
        selected, metric = packet["selected_frames"], packet["hand_joints"]
    palm_raw = np.linalg.norm(world[selected, 5]-world[selected, 17], axis=-1)*1000
    palm_metric = np.linalg.norm(metric[selected, 5]-metric[selected, 17], axis=-1)*1000
    quality = dict(selected_frames=selected.tolist(),
                   raw_estimated_palm_width_mm=palm_raw.tolist(), aligned_palm_width_mm=palm_metric.tolist(),
                   raw_palm_width_min_max_mm=[float(palm_raw.min()), float(palm_raw.max())],
                   aligned_palm_width_min_max_mm=[float(palm_metric.min()), float(palm_metric.max())],
                   interpretation="Same hand should not change palm width this much: detection coverage is not 3D accuracy. This chain fails a precision-demonstration quality requirement.")
    (report/"perception_quality.json").write_text(json.dumps(quality, indent=2)+"\n")
    figure_results(report, groups, curves, bc_losses)
    lines = ["# 从圆柱原始视频重做：完整实验结果", "", "## 完成范围", "",
             "从新的原始 RGB 视频重新提取到策略训练、评估的链路已执行。9 次新初始化训练、66 项评估、4224 个正式回合。",
             "完成链路不等于成功抓取，也不等于复现原论文完整感知系统。输入只有一个视频，姿态及示范质量较差。", "",
             "[逐步说明与术语](../../../docs/experiments/cylinder_raw_video_20261009.md) | [文件索引](../README.md)", "",
             "## 固定预算结果", "", "以下是655360在线步结果；仅BC、仅参考无在线训练。学习方法各3个种子，每个种子64个配对初态。", "",
             "| 方法 | XY ±2mm | XY ±6mm | 加0.15N推力 |", "|---|---:|---:|---:|"]
    for method in ("ppo", "bc", "guided", "zero", "residual"):
        budget = 0 if method in ("bc", "zero") else 655360
        values = [next(g["mean_success"] for g in groups if (g["condition"],g["method"],g["budget"]) == (c,method,budget))
                  for c in ("B0_nominal", "B1_position", "B5_push")]
        lines.append("| "+ZH[method]+" | "+" | ".join(f"{100*v:.2f}%" for v in values)+" |")
    if not any(row["successes"] for row in runs):
        lines += ["", "**本轮所有正式测试的清洁30秒成功率均为零，没有训练出达标的抓取策略。**",
                  "这是一组完整执行后的负结果，不是成功演示，也不支持视频引导优于纯PPO的结论。"]
    lines += ["", "## 不只看平均数", "", "| 方法 | 51001 | 51002 | 51003 |", "|---|---:|---:|---:|"]
    for method in ("ppo", "guided", "residual"):
        values = [next(r["clean_success30"] for r in runs if (r["method"],r["train_seed"],r["budget"],r["condition"])
                      == (method,seed,655360,"B0_nominal")) for seed in (51001,51002,51003)]
        lines.append("| "+ZH[method]+" | "+" | ".join(f"{100*v:.2f}%" for v in values)+" |")
    lines += ["", "## 失败发生在哪里", "",
              "正常位置、固定最终预算。原始成功不包含子步穿透/禁止接触验收，不能当作清洁成功。", "",
              "| 方法 | 原始15秒 | 原始30秒 | 平均最长保持秒 | 掉落率 | 撤托后禁止接触率 | 撤托后穿透违规率 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for method in ("ppo", "bc", "guided", "zero", "residual"):
        g = next(g for g in groups if (g["condition"],g["method"],g["budget"])
                 == ("B0_nominal",method,0 if method in ("bc","zero") else 655360))
        lines.append(f"| {ZH[method]} | {g['mean_raw_success15']*100:.2f}% | {g['mean_raw_success30']*100:.2f}% | "
                     f"{g['mean_hold']:.2f} | {g['drop_rate']*100:.2f}% | {g['forbidden_rate']*100:.2f}% | {g['depth_failure_rate']*100:.2f}% |")
    raw15 = [e for e in episodes if e["success15"]]
    raw15_sources = "; ".join(f"{ZH[m]}，训练种子{s}" for m, s in
                              sorted({(e["method"], e["train_seed"]) for e in raw15})) or "无"
    lines += ["", f"全部4224次正式测试中，有{len(raw15)}次达到原始15秒指标，"
              "详细方法、训练种子和条件见episodes.csv。"
              "它们不能直接合并成某个方法的成功率，也不能把相同训练种子在不同条件的测试当作独立训练重复。",
              f"本轮原始15秒成功的来源：{raw15_sources}。短时成功不等于稳定30秒抓持。",
              "", "## 怎样理解", "",
              "1. 旧圆柱实验的高成功率来自方块参考及仿真校正；不能作为本次圆柱原始视频链路的结果。",
              "2. 本次重新识别没有漏帧，但单目深度、遮挡和左右手映射仍有误差；指尖相对视觉估计目标的平均拟合残差为23–36mm。"
              "这不是相对三维真值测量的视觉精度，本视频没有提供独立真值。",
              f"所用31帧中，视觉模型估计掌宽为{palm_raw.min():.1f}–{palm_raw.max():.1f}mm，对齐后为{palm_metric.min():.1f}–{palm_metric.max():.1f}mm。"
              "同一只手的掌宽不应如此变化，说明它未达到精准三维示范的质量要求，不能只凭检测率放行。",
              "3. 未经成功率校正的新参考，32次回放均在早期失败，只提供257个终止前样本。BC训练的不是成功专家。",
              "BC的验证损失可以降得很低，但这仅说明它学会重复这批失败初段动作，不表示学会抓握。不能把低模仿误差当作任务成功。",
              "4. 残差每步最多修正0.08rad。它保留视频主导这一约束，也因此不能自由替换一个偏差很大的基础动作。",
              "5. PPO仍通过相同奖励交互学习。本轮没有按结果调奖励、增大修正范围或换成功参考。",
              "6. 这是低质量先验下的完整固定配方实验，不足以给高质量视频引导或充分调参PPO下普遍结论。", "",
              "先验质量不足是直接可见的问题，但未证明它是失败的唯一原因。物体尺寸、固定手底座的可达性、奖励和预算也可能影响结果，"
              "本轮没有做因果消融来分离这些因素，因此不把推测写成已证明结论。", "",
              "推力在10秒时施加。若策略在此之前就未形成有效抓持，推力条件的低成功率不能单独证明抗扰动差；"
              "runs.csv额外记录施扰前及施扰后有效抓持比例，避免把未抓住与扰动导致失败混为一谈。", "",
              "## 图表", "", "![固定预算成功率](success_rates.png)", "", "![完整学习曲线](learning_curves.png)", "",
              "![重定向误差与参考关节](retarget_diagnostics.png)", "",
              "![BC训练与验证损失](bc_losses.png)", "",
              "成功率曲线用于判断任务是否完成；奖励曲线仅作诊断，不能替代抓取成功。阴影是种子标准差，不是置信区间。", "",
              "## 统计边界", "",
              "每个种子的64次测试只衡量该固定模型在初态分布上的表现，不能当作64次独立训练。",
              "runs.csv给出每个固定模型的精确二项式95%区间；例如0/64仍有约5.60%的上界，不代表总体成功概率严格等于零。",
              "三种子配对符号翻转检验最小双侧p值为0.25；不能凭本轮写p<0.05的普遍优越性结论。",
              "仅参考没有训练种子，因此不复制成三次训练。4224是所有方法/预算/条件测试的总次数，不是单一方法独立样本量。", "",
              "## 全量文件", "", "- [逐回合数据](episodes.csv)", "- [逐种子/预算/条件汇总及条件置信区间](runs.csv)",
              "- [分组均值与种子标准差](groups.csv)", "- [学习曲线原始数值](learning_curves.csv)",
              "- [BC训练/验证损失](bc_losses.csv)",
              "- [配对差值及PPO相对自身非RL对照的贡献](paired_differences.json)", "- [完整矩阵审计](../audit.json)", "",
              "所有回合评分和配置已核查，另重算444条已保存完整预览轨迹；未声称保存并重算全部4224条状态轨迹。"]
    (report/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(audit), flush=True)


def figure_results(report, groups, curves, bc_losses):
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    methods = ("ppo", "bc", "guided", "zero", "residual")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, condition in zip(axes, ("B0_nominal", "B1_position", "B5_push")):
        values = [next(g["mean_success"] for g in groups if (g["condition"],g["method"],g["budget"])
                       == (condition,m,0 if m in ("bc","zero") else 655360))*100 for m in methods]
        deviations = [next(g["seed_std"] or 0. for g in groups if (g["condition"],g["method"],g["budget"])
                           == (condition,m,0 if m in ("bc","zero") else 655360))*100 for m in methods]
        bars = ax.bar(np.arange(5), values, yerr=deviations, capsize=3, color=[COLORS[m] for m in methods])
        ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
        ax.set_xticks(np.arange(5), [LABELS[m].replace(" + ","\n+ ") for m in methods], rotation=25, ha="right")
        ax.set_ylim(0, 105)
        ax.set_title(condition)
        ax.set_ylabel("Clean 30-second success (%)")
    fig.savefig(report/"success_rates.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for method in ("ppo", "guided", "residual"):
        for ax, field, label in zip(axes, ("clean_success30","mean_hold","mean_return"),
                                    ("Clean success (%)","Mean longest hold (s)","Episode return (not success)")):
            steps = np.array(sorted({r["step"] for r in curves if r["method"] == method}))
            values = np.array([[r[field] for r in curves if r["method"] == method and r["step"] == step] for step in steps])
            if field == "clean_success30":
                values *= 100
            mean, std = values.mean(1), values.std(1, ddof=1)
            ax.plot(steps, mean, label=LABELS[method], color=COLORS[method], marker=".")
            ax.fill_between(steps, mean-std, mean+std, color=COLORS[method], alpha=.12)
            ax.set_xlabel("Online environment steps")
            ax.set_ylabel(label)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    axes[0].set_ylim(-1,101)
    axes[2].legend(fontsize=8)
    fig.savefig(report/"learning_curves.png", dpi=160)
    plt.close(fig)
    with np.load(OUT/"retarget_neutral/reference.npz") as ref:
        times, errors, ctrl = ref["time"], ref["errors_m"]*1000, ref["ctrl"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for i, finger in enumerate(("Thumb","Index","Middle","Ring","Pinky")):
        axes[0].plot(times, errors[:,i], label=finger)
    axes[0].set(xlabel="Retarget segment time (s)", ylabel="Fingertip fitting error (mm)")
    axes[0].legend(fontsize=8)
    for i, label in ((11,"Index MCP"),(12,"Index PIP"),(13,"Thumb CMC"),(16,"Thumb PIP")):
        axes[1].plot(times, ctrl[:,i], label=label)
    axes[1].set(xlabel="Retarget segment time (s)", ylabel="Reference joint target (rad)")
    axes[1].legend(fontsize=8)
    fig.savefig(report/"retarget_diagnostics.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for seed in (51001, 51002, 51003):
        rows = [row for row in bc_losses if row["train_seed"] == seed]
        line = ax.semilogy([r["step"] for r in rows], [r["train_loss"] for r in rows], label=f"{seed} train")[0]
        ax.semilogy([r["step"] for r in rows], [r["validation_loss"] for r in rows],
                    linestyle="--", color=line.get_color(), label=f"{seed} validation")
    ax.set(xlabel="BC optimizer updates", ylabel="Imitation loss (not grasp success)")
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(report/"bc_losses.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
