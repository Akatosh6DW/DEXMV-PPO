"""Summarize completed S10 tests, retaining failures and paired uncertainty."""

import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR","/tmp/s10_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from s10_mobile_base import OUT
from run_s10_study import check


def write_csv(path,rows):
    with path.open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def wilson(successes,count):
    p=successes/count; z=1.96; scale=1+z*z/count
    center=(p+z*z/(2*count))/scale
    width=z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/scale
    return [float(center-width),float(center+width)]


def bootstrap(diff,seed=217):
    """Resample training origins and common states, never pool them as independent."""
    rng=np.random.default_rng(seed)
    values=[]
    for _ in range(10000):
        origins=rng.integers(0,len(diff),len(diff))
        states=rng.integers(0,diff.shape[1],diff.shape[1])
        values.append(diff[origins][:,states].mean())
    return np.quantile(values,[.025,.975]).tolist()


def main():
    protocol=check()
    report=OUT/"report"; report.mkdir(exist_ok=True)
    episodes=[]; models=[]; arrays={}
    for method in ("reference","reference_exact","ppo","bc","guided","residual"):
        per_origin=[]
        for origin in ([0] if method.startswith("reference") else protocol["seeds"]):
            folder_method="reference_exact_quality" if method=="reference_exact" else method
            run=OUT/"eval"/f"{folder_method}_s{origin}"
            summary=json.loads((run/"summary.json").read_text())
            config=json.loads((run/"run_config.json").read_text())
            assert summary["status"]=="completed" and summary["final_step"]==0
            assert [config["eval_start"],config["eval_count"]]==protocol["test"]
            rows=json.loads((run/"step_000000000/episodes.json").read_text())
            if method=="reference_exact":
                assert all(r["residual_rms_rad"]==0 for r in rows)
                old=json.loads((OUT/"eval/reference_exact_s0/step_000000000/episodes.json").read_text())
                assert len(old)==len(rows)
                for before,after in zip(old,rows):
                    assert all(after[k]==v for k,v in before.items()),"Quality wrapper changed original evaluation"
            assert [r["seed"] for r in rows]==list(range(protocol["test"][0],sum(protocol["test"])))
            episodes += [dict(method=method,training_seed=origin,**row) for row in rows]
            successes=np.array([r["success30"] for r in rows],float)
            per_origin.append(successes)
            passed=[r for r in rows if r["success30"]]
            models.append(dict(method=method,training_seed=origin,count=len(rows),successes=int(successes.sum()),
                success_rate=float(successes.mean()),mean_hold_sec=float(np.mean([r["hold_sec"] for r in rows])),
                full_path_success_rate=float(np.mean([r["full_path_success30"] for r in rows])),
                peak_force_n=max(r["peak_force_n"] for r in rows),
                success_peak_force_n=max((r["peak_force_n"] for r in passed),default=0.),
                success_force_p95_n=float(np.quantile([r["peak_force_n"] for r in passed],.95)) if passed else 0.,
                max_penetration_m=max(r["max_released_penetration_m"] for r in rows),
                max_path_penetration_m=max(r["max_path_penetration_m"] for r in rows),
                max_tracking_error_m=max(r["max_tracking_error_m"] for r in rows),
                residual_rms_rad=float(np.mean([r["residual_rms_rad"] for r in rows]))))
        arrays[method]=np.array(per_origin)
    write_csv(report/"episodes.csv",episodes)
    write_csv(report/"models.csv",models)
    failure_rows=[]
    for row in models:
        group=[r for r in episodes if r["method"]==row["method"] and r["training_seed"]==row["training_seed"]]
        failed=[r for r in group if not r["success30"]]
        failure_rows.append(dict(method=row["method"],training_seed=row["training_seed"],failures=len(failed),
            early_termination=sum(r["duration_sec"]<42 for r in failed),
            long_hold_but_invalid_end=sum(r["hold_sec"]>=30 and not r["final_valid"] for r in failed),
            below_30s=sum(r["hold_sec"]<30 for r in failed),
            released_penetration=sum(r["max_released_penetration_m"]>.003 for r in failed),
            forbidden_support=sum(r["forbidden"] for r in failed)))
    write_csv(report/"failure_gates.csv",failure_rows)
    comparisons={}
    for left,right in (("guided","bc"),("residual","reference_exact"),("residual","reference"),
                       ("guided","ppo"),("residual","ppo")):
        delta=arrays[left]-arrays[right]
        comparisons[f"{left}_minus_{right}"]=dict(mean=float(delta.mean()),ci95=bootstrap(delta),
            per_origin=delta.mean(1).tolist(),gained=(delta>0).sum(1).tolist(),lost=(delta<0).sum(1).tolist())
    statistics=dict(policy_comparisons=comparisons,interpretation="Only three training origins; paired descriptive bootstrap, not broad robustness proof")
    control_summary=json.loads((OUT/"alignment_controls/summary.json").read_text())
    assert control_summary["status"]=="completed"
    controls={}
    for row in control_summary["groups"]:
        group=row["group"]
        controls[group]=json.loads((OUT/"alignment_controls"/f"{group}_episodes.json").read_text())
        assert len(controls[group])==128
        assert [r["seed"] for r in controls[group]]==list(range(230000,230128))
        row["wilson95"]=wilson(row["successes"],row["count"])
        successes=[r for r in controls[group] if r["success30"]]
        row["success_peak_force_n"]=max((r["peak_force_n"] for r in successes),default=0.)
    mobile=np.array([r["success30"] for r in controls["aligned_moving"]],float)
    statistics["alignment_comparisons"]={group:dict(mean=float((mobile-np.array([r["success30"] for r in rows])).mean()),
        ci95=bootstrap((mobile-np.array([r["success30"] for r in rows]))[None,:]))
        for group,rows in controls.items() if group!="aligned_moving"}
    common_force=np.array([(a["peak_force_n"],b["peak_force_n"])
        for a,b in zip(controls["fixed_template"],controls["aligned_moving"])
        if a["success30"] and b["success30"]])
    statistics["common_success_contact_force"]=dict(count=len(common_force),
        columns=["fixed_template","aligned_moving"],median_n=np.median(common_force,axis=0).tolist(),
        mean_n=common_force.mean(0).tolist(),moving_lower_count=int((common_force[:,1]<common_force[:,0]).sum()),
        scope="Post-hoc description conditioned on both succeeding; not all states or hardware safety")
    (report/"statistics.json").write_text(json.dumps(statistics,indent=2)+"\n")
    labels={"ppo":"Aligned + scratch PPO","bc":"Aligned + BC","guided":"Aligned + BC then PPO",
            "residual":"Aligned + residual PPO","reference":"Reference (normalized)",
            "reference_exact":"Reference (zero residual)"}
    order=["ppo","bc","guided","residual","reference","reference_exact"]
    fig,ax=plt.subplots(figsize=(10,5))
    ax.bar(range(6),[100*arrays[k].mean() for k in order],color=["#777777","#c79c36","#3c876a","#ad465d","#4c88ac","#74a9ba"])
    for i,k in enumerate(order):
        offset=np.linspace(-.06,.06,len(arrays[k])) if len(arrays[k])>1 else np.zeros(1)
        ax.scatter(i+offset,100*arrays[k].mean(1),color="black",s=25,zorder=3)
    ax.set_xticks(range(6),[labels[k] for k in order],rotation=15,ha="right")
    ax.set_ylabel("Strict 30-second success (%)"); ax.set_ylim(-2,105)
    ax.set_title("S10 held-out policy results; dots are training origins")
    fig.tight_layout(); fig.savefig(report/"policy_success.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    training_rows=[]
    for method,color in (("ppo","#777777"),("guided","#3c876a"),("residual","#ad465d")):
        series=[]
        for origin in protocol["seeds"]:
            with (OUT/"train"/f"{method}_s{origin}"/"metrics.csv").open() as f:
                data=list(csv.DictReader(f))
            x=np.array([int(r["step"]) for r in data])
            y=np.array([float(r["success30"])*100 for r in data])
            summary=json.loads((OUT/"train"/f"{method}_s{origin}"/"summary.json").read_text())
            assert summary["status"]=="completed" and summary["final_step"]==protocol["steps"]
            training_rows.append(dict(method=method,training_seed=origin,**summary["success30"],
                final_hold_sec=summary["hold_sec"]["last"],
                final_peak_force_n=float(data[-1]["peak_force_n"]),
                final_max_penetration_m=float(data[-1]["max_penetration_m"])))
            series.append(y); ax.plot(x,y,color=color,alpha=.2)
        ax.plot(x,np.mean(series,axis=0),color=color,label=labels[method])
    ax.set_xlabel("Online environment interactions"); ax.set_ylabel("Development success (%)")
    ax.legend(); fig.tight_layout(); fig.savefig(report/"training_curves.png",dpi=180); plt.close(fig)
    write_csv(report/"training_runs.csv",training_rows)
    lines=["# S10：整手对准抓圆柱结果", "", "## 读表前先看", "",
        "本实验让整手从共同起点实际移动、转向，再抓取。基座为理想笛卡尔控制器驱动的六自由度动态刚体，不是真实机械臂。",
        "四组共享同一对准器；从零 PPO 指手指网络从零训练，不是整个系统没有视频或几何先验。",
        "本轮未重新提取原始视频；使用已有 DexMV 重定向并校正的示范。",
        "相对 S09 增加两秒接近时间，并抬高工作空间25厘米，不能直接把两次实验成功率相减归因于某个单一改动。", "",
        "[通俗结论与失败分析](../../../../docs/experiments/S10/结论与分析.md)；[从头理解整个流程](../../../../docs/experiments/S10/从头理解整手对准实验.md)。", "",
        "## 独立策略测试", "", "| 方法 | 三来源成功率 | 平均 |", "|---|---|---|"]
    names={"ppo":"对齐＋从零手指PPO","bc":"对齐＋仅BC","guided":"对齐＋BC后PPO",
           "residual":"对齐＋参考加残差PPO","reference":"对齐＋参考回放（完整动作归一化）",
           "reference_exact":"对齐＋参考回放（严格零残差）"}
    for k in order:
        lines.append(f"| {names[k]} | {' / '.join(f'{100*x:.2f}%' for x in arrays[k].mean(1))} | {100*arrays[k].mean():.2f}% |")
    # Describe held-out failures by geometry; never use these bins to retune policies.
    from s10_mobile_base import MobileCylinder
    sim = MobileCylinder((OUT/"inputs/model.xml").read_text())
    with np.load(OUT/"eval/reference_exact_quality_s0/step_000000000/initial_states.npz") as initial:
        quaternion = initial["qpos"][:,sim.oid+3:sim.oid+7]
    axis = Rotation.from_quat(quaternion[:,[1,2,3,0]]).apply(np.tile([0.,-1.,0.],(len(quaternion),1)))
    tilt = np.rad2deg(np.arccos(np.clip(np.abs(axis[:,2]),0,1)))
    bins = np.minimum((tilt//30).astype(int),2)
    orientation_rows=[]
    orientation_matrix=[]
    for method in order:
        rates=[]
        for index in range(3):
            mask=bins==index
            assert mask.any()
            values=arrays[method][:,mask].mean(1)
            rates.append(float(values.mean()))
            for origin,value in zip(([0] if method.startswith("reference") else protocol["seeds"]),values):
                orientation_rows.append(dict(method=method,training_seed=origin,
                    axis_tilt_degrees=f"{index*30}-{(index+1)*30}",unique_states=int(mask.sum()),success_rate=float(value)))
        orientation_matrix.append(rates)
    write_csv(report/"orientation_groups.csv",orientation_rows)
    fig,ax=plt.subplots(figsize=(9,5))
    matrix=100*np.array(orientation_matrix)
    heat=ax.imshow(matrix,vmin=0,vmax=100,cmap="viridis",aspect="auto")
    ax.set_xticks(range(3),[f"{i*30}-{(i+1)*30} degrees (n={(bins==i).sum()})" for i in range(3)])
    ax.set_yticks(range(len(order)),[labels[k] for k in order])
    for i in range(len(order)):
        for j in range(3):
            ax.text(j,i,f"{matrix[i,j]:.1f}%",ha="center",va="center",color="black" if matrix[i,j]>60 else "white")
    ax.set_xlabel("Initial cylinder axis tilt from vertical; mean across training origins")
    fig.colorbar(heat,ax=ax,label="Strict success (%)")
    fig.tight_layout();fig.savefig(report/"orientation_success.png",dpi=180);plt.close(fig)
    lines += ["", "每个训练来源测试相同的128个初态；每个学习方法只有3个独立训练来源。参考回放只评估一次，不伪装成3次独立训练。", "",
              "逐例hold_sec及汇总mean_hold_sec指一回合中最长的一段连续合格保持，不是结束时的连续保持，也不是累计所有断续接触时间。曾保持超过30秒但结束不合格仍判失败。", "",
              "## 配对差异", "", "| 比较 | 平均差（百分点） | 描述性95%区间 |", "|---|---:|---|"]
    for key,value in comparisons.items():
        left,right=key.split("_minus_")
        lines.append(f"| {names[left]} 减 {names[right]} | {100*value['mean']:+.2f} | [{100*value['ci95'][0]:+.2f}, {100*value['ci95'][1]:+.2f}] |")
    lines += ["", "## 单独的几何对照", "", "这是另一批128个配对初态，不与上表交叉相减。", "",
              "| 控制 | 成功数 | 成功率 | 全部案例峰值力（N） | 成功案例峰值力（N） |", "|---|---:|---:|---:|---:|"]
    control_names=dict(fixed_template="固定手、选定单参考",fixed_bank="固定手、原六参考选择",
        aligned_preplaced="直接预摆到对准位置",aligned_moving="从共同起点实际移动对准")
    for r in control_summary["groups"]:
        lines.append(f"| {control_names[r['group']]} | {r['successes']}/{r['count']} | {100*r['success30']:.3f}% | {r['max_force']:.2f} | {r['success_peak_force_n']:.2f} |")
    lines += ["", "## 全过程质量", "", "| 方法 | 来源 | 原成功率 | 全程穿透≤3mm成功率 | 成功案例最大力N | 最大路径穿透mm |", "|---|---:|---:|---:|---:|---:|"]
    for r in models:
        lines.append(f"| {names[r['method']]} | {r['training_seed']} | {100*r['success_rate']:.2f}% | {100*r['full_path_success_rate']:.2f}% | {r['success_peak_force_n']:.2f} | {1000*r['max_path_penetration_m']:.2f} |")
    lines += ["", "## 原始文件", "", "- `episodes.csv`：所有策略、所有来源、所有测试初态，包含失败。",
              "- `models.csv`：成功率、保持时长、峰值力、穿透和残差幅度。",
              "- `training_runs.csv`：9次训练的开发集first/last/max/best_step及末步接触质量；不把开发集最高点冒充最终测试。",
              "- `failure_gates.csv`：提前终止、曾长时保持但结束不合格、未连续保持30秒等多标签失败分类；列之间可能重叠。",
              "- `statistics.json`：共同初态配对差异、重采样区间及新增/丢失成功。",
              "- `policy_success.png`、`training_curves.png`：最终成功率与训练曲线。",
              "- `orientation_groups.csv`、`orientation_success.png`：圆柱轴相对竖直方向的倾角分组；只作事后描述，不据此回调参数。0度为竖直，90度为水平，不是手腕转角。",
              "- `../alignment_controls/`：四种几何控制的逐例原始数据。",
              "", "成功不等于安全：必须结合峰值力和穿透看结果；没有实机验证。"]
    (report/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(status="completed",models=models,statistics=statistics)),flush=True)


if __name__=="__main__":
    main()
