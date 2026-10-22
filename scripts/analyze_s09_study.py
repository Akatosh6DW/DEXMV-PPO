"""Analyze the COMPLETE frozen S09 matrix without pooling training replicates away."""

import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/s09-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from retarget_s09_reference import digest
from run_s09_study import METHODS, OUT, ROOT, SEEDS, STEPS, STRATA, check_protocol

LABELS={"ppo":"Pure PPO","guided":"BC + PPO","residual":"Reference + residual PPO","reference":"Reference replay"}
CHINESE={"ppo":"纯PPO","guided":"BC后PPO","residual":"参考＋残差PPO","reference":"参考回放"}


def interval(matrix, seed=19001):
    """Descriptive hierarchical bootstrap over training seeds and paired test states."""
    rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(2000):
        model=rng.integers(len(matrix),size=len(matrix))
        episode=rng.integers(matrix.shape[1],size=matrix.shape[1])
        samples.append(matrix[np.ix_(model,episode)].mean())
    return np.quantile(samples,[.025,.975]).tolist()


def main():
    protocol=check_protocol()
    raw=[]
    summaries=[]
    matrices_by_group={}
    jobs=0
    for stratum,(start,count) in STRATA.items():
        paired_initial=None
        for method in (*METHODS,"reference"):
            matrices=[]
            for seed in ((0,) if method=="reference" else SEEDS):
                name="reference" if method=="reference" else f"{method}_s{seed}"
                run=OUT/"eval"/stratum/name
                summary=json.loads((run/"summary.json").read_text())
                assert summary["status"]=="completed" and summary["final_step"]==0
                config=json.loads((run/"run_config.json").read_text())
                assert config["orientation"]==stratum and config["eval_start"]==start and config["eval_count"]==count
                assert config["model_sha256"]==digest(OUT/"inputs/model.xml")
                assert config["reference_sha256"]==digest(OUT/"inputs/reference.npz")
                for source,expected in protocol["source_hashes"].items():
                    if source.startswith("scripts/") and not source.endswith("run_s09_study.py"):
                        assert config["sources"][source.split("/")[-1]]==expected
                assert config["approach_weight"]==2 and config["coverage_weight"]==2
                if method!="reference":
                    train=OUT/"train"/name
                    training=json.loads((train/"summary.json").read_text())
                    recipe=json.loads((train/"run_config.json").read_text())
                    assert training["status"]=="completed" and training["final_step"]==STEPS
                    assert recipe["method"]==method and recipe["seed"]==seed and recipe["steps"]==STEPS
                    assert recipe["learning_rate"]==protocol["recipes"][method]["learning_rate"]
                    assert recipe["std"]==protocol["recipes"][method]["std"]
                    assert recipe["approach_weight"]==2 and recipe["coverage_weight"]==2
                    for source,expected in protocol["source_hashes"].items():
                        if source.startswith("scripts/") and not source.endswith("run_s09_study.py"):
                            assert recipe["sources"][source.split("/")[-1]]==expected
                    if method=="guided":
                        assert recipe["bc_updates"]==4000 and recipe["dataset_sha256"]==digest(OUT/"demonstrations/trajectories.npz")
                    assert config["checkpoint_sha256"]==digest(train/f"step_{STEPS:09d}/params.pkl")
                folder=run/"step_000000000"
                with np.load(folder/"initial_states.npz") as z:
                    np.testing.assert_array_equal(z["seeds"],np.arange(start,start+count))
                    if paired_initial is None:
                        paired_initial=z["qpos"].copy()
                    else:
                        np.testing.assert_array_equal(z["qpos"],paired_initial)
                episodes=json.loads((folder/"episodes.json").read_text())
                assert len(episodes)==count and [e["seed"] for e in episodes]==list(range(start,start+count))
                for e in episodes:
                    assert e["released_rope_force_n"]==0
                    if e["success30"]:
                        assert e["success15"] and e["hold_sec"]>=30 and e["final_valid"]
                        assert not e["forbidden"] and e["max_released_penetration_m"]<=.003 and e["initial_penetration_m"]<=.001
                    raw.append(dict(stratum=stratum,method=method,training_seed=seed,**e))
                matrices.append([e["success30"] for e in episodes])
                jobs+=1
            matrix=np.asarray(matrices,float)
            matrices_by_group[stratum,method]=matrix
            summaries.append(dict(stratum=stratum,method=method,success_rate=float(matrix.mean()),
                per_training_seed=matrix.mean(1).tolist(),episodes_per_model=count,
                training_replicates=0 if method=="reference" else len(SEEDS),descriptive_ci95=interval(matrix)))
    assert jobs==50 and len(raw)==2560
    output=OUT/"report"
    output.mkdir(exist_ok=True)
    with (output/"episodes.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
    (output/"summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    differences=[]
    for stratum in STRATA:
        for baseline in ("ppo","guided","reference"):
            delta=matrices_by_group[stratum,"residual"]-matrices_by_group[stratum,baseline]
            differences.append(dict(stratum=stratum,contrast="residual minus "+baseline,
                success_rate_difference=float(delta.mean()),descriptive_ci95=interval(delta)))
    (output/"paired_differences.json").write_text(json.dumps(differences,indent=2)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout="constrained")
    colors=("#777777","#3c8c70","#d25c5c","#4a86b7")
    for i,method in enumerate((*METHODS,"reference")):
        rows=[r for r in summaries if r["method"]==method]
        axes[0].bar(np.arange(5)+(i-1.5)*.19,[r["success_rate"]*100 for r in rows],.19,label=LABELS[method],color=colors[i])
        r=next(r for r in rows if r["stratum"]=="uniform")
        axes[1].scatter([i]*len(r["per_training_seed"]),np.array(r["per_training_seed"])*100,color=colors[i],s=40)
        lo,hi=np.array(r["descriptive_ci95"])*100
        axes[1].plot([i,i],[lo,hi],color=colors[i],lw=2)
    axes[0].set(xticks=np.arange(5),xticklabels=list(STRATA),ylabel="Strict 30s success (%)",ylim=(0,100),title="Matched initial states, frozen R0")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=8)
    axes[1].set(xticks=np.arange(4),xticklabels=["PPO","BC+PPO","Residual","Reference"],ylim=(0,100),ylabel="Success (%)",title="Uniform SO(3): training-seed results")
    fig.savefig(output/"success_rates.png",dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout="constrained")
    curves=[]
    for ax,method,color in zip(axes,METHODS,colors):
        for seed in SEEDS:
            run=OUT/"train"/f"{method}_s{seed}"
            with (run/"metrics.csv").open() as f:
                records=list(csv.DictReader(f))
            steps=[int(r["step"]) for r in records]
            rates=[100*float(r["success30"]) for r in records]
            ax.plot(steps,rates,marker="o",label=str(seed),alpha=.7)
            curves.extend(dict(method=method,training_seed=seed,**r) for r in records)
        ax.set(title=LABELS[method],xlabel="Online transitions",ylabel="Development success (%)",ylim=(0,100))
        ax.legend(title="Training seed",fontsize=8)
    fig.savefig(output/"learning_curves.png",dpi=160)
    plt.close(fig)
    with (output/"learning_curves.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    offline=[]
    for stage in ("P01_source_ik","P02_bounded_correction","P03_long_selection"):
        root=OUT/stage
        if stage=="P01_source_ik":
            samples=json.loads((root/"development.json").read_text())
        elif stage=="P02_bounded_correction":
            samples=[]
            for folder in sorted(root.glob("pose_*")):
                samples += json.loads((folder/"search.json").read_text())
                samples.append(json.loads((folder/"report.json").read_text())["validation"])
        else:
            samples=[]
            for folder in sorted(root.glob("pose_*")):
                selection=json.loads((folder/"selection.json").read_text())
                for item in selection["candidates"]:
                    samples += [item["nominal"],*item.get("jittered_development",[])]
                samples.append(selection["candidates"][selection["best"]]["nominal"])
        offline.append(dict(stage=stage,rollouts=len(samples),
            active_control_transitions=sum(round(r["duration_sec"]/.05) for r in samples)))
    (output/"offline_cost.json").write_text(json.dumps(dict(reference_preparation=offline,
        ik_endpoint_starts=36,ik_sampled_frame_fits=96,bc_updates_per_guided_seed=4000,
        collection_attempts=128,collection_successes=56,
        collection_active_control_transitions=sum(round(r["duration_sec"]/.05) for r in
            json.loads((OUT/"demonstrations/step_000000000/episodes.json").read_text())),
        note="Active rollout transitions, not complete GPU kernel counts; IK function calls and total BC wall time were not instrumented."),indent=2)+"\n")
    lines=["# S09固定奖励四组实验结果","", "本报告由完整50项评估、2560个回合生成，缺项时脚本会拒绝发布。",
        "三种学习方法各3个训练种子，每次163840步；参考没有训练种子。以下不是把每个回合当作独立训练。","",
        "## 结果","","| 朝向 | 方法 | 平均成功率 | 各训练种子成功率 |","|---|---|---:|---|"]
    for r in summaries:
        values=" / ".join(f"{v*100:.2f}%" for v in r["per_training_seed"])
        lines.append(f"| {r['stratum']} | {CHINESE[r['method']]} | {100*r['success_rate']:.2f}% | {values} |")
    lines += ["","![四组成功率](success_rates.png)","","## 学习过程","",
        "![开发监测曲线](learning_curves.png)","这些曲线仅用8个固定开发初态，不能代替最终测试。BC后PPO的0步是BC完成后的起点；残差的0步已经有视频参考，不是从零动作开始。",
        "监测elapsed_sec不含BC预训练；离线校正与采集另外统计，不能只用PPO步数比较全部成本。",
        "", "## 结果边界","",
        "仅有3个学习种子，区间为描述性分层bootstrap，不作强显著性宣称。参考组区间只反映测试初态抽样，不能解释为训练随机性。",
        "若某组样本全成功/全失败，经验bootstrap区间可能退化成单点；这不代表真实成功概率已确定为100%或0%，不能外推为未来必然成功/失败。",
        "圆柱朝向覆盖完整SO(3)采样域，不表示每个方向都抓得到。物体位置限于固定手的可达区域；悬吊为通过质心的理想平衡吊带，不是端部绑绳或真实柔性绳接触模型。",
        "示范是已有DexMV处理后指尖轨迹经IK与仿真校正生成，不是新拍摄的多视角视频；只选成功回合作BC数据，但全部参考尝试和全部测试失败均保留。",
        "相同在线步数不等于总成本相同。IK、物理参考校正、示范采集与BC有额外开销。部分失败存在过大穿透和力峰值，不能直接视为硬件可部署。",
        "本表固定奖励版本R0，不混入之后的奖励调参。主指标要求最后仍合格；中途拿住30秒后掉落不能算成功。",
        "","- [逐回合原始数据](episodes.csv)","- [分层统计与描述性区间](summary.json)",
        "- [配对成功率差值](paired_differences.json)","- [逐种子学习曲线数据](learning_curves.csv)",
        "- [额外离线计算量](offline_cost.json)"]
    (output/"results.md").write_text("\n".join(lines)+"\n")
    (output/"matrix_verification.json").write_text(json.dumps(dict(status="matrix_verified",jobs=jobs,episodes=len(raw),
        paired_initial_states=True,checkpoint_binding=True,protocol_sha256=digest(OUT/"protocol.json"),
        warning="Matrix and metric consistency audit, not yet an independent replay of every saved physical trace"),indent=2)+"\n")
    print(json.dumps(dict(jobs=jobs,episodes=len(raw),output=str(output))),flush=True)


if __name__=="__main__":
    main()
