"""Cluster-paired intervals and artifact checks for frozen-noise diagnostics."""

import csv
import json
from pathlib import Path

import numpy as np

from analyze_s09_bc_confirmation import write_csv
from retarget_s09_reference import digest


def main():
    root=Path(__file__).resolve().parents[1]
    folder=root/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018/D42_temporal_noise"
    read=lambda path:json.loads(path.read_text())
    summary=read(folder/"summary.json")
    assert summary["status"]=="completed_diagnostic_only" and not summary["smoke"]
    assert summary["steps"]==800 and summary["policy_updates"]==0 and len(summary["seeds"])==32
    assert summary["checkpoint_unchanged"]
    assert digest(root/"artifacts/orca_benchmark_suite/S09/bc_repair/I27_phase_s13102/step_000000000/params.pkl")==summary["checkpoint_sha256"]
    for name,field in (("model.xml","model_sha256"),("reference.npz","reference_sha256")):
        assert digest(root/"artifacts/orca_benchmark_suite/S09/inputs"/name)==summary[field]
    for name,sha in summary["sources"].items():
        assert digest(folder/"sources"/name)==sha
    with (folder/"episodes.csv").open() as stream:
        episodes=list(csv.DictReader(stream))
    with (folder/"audits.csv").open() as stream:
        audits=list(csv.DictReader(stream))
    assert len(episodes)==len(audits)==160
    keyed={(r["arm"],int(r["draw"]),int(r["seed"])):r for r in episodes}
    assert len(keyed)==160
    audit_keys={(r["arm"],int(r["draw"]),int(r["seed"])) for r in audits}
    assert audit_keys==set(keyed)
    physical=0
    lost_details=[]
    for row in audits:
        key=row["arm"],int(row["draw"]),int(row["seed"])
        path=folder/key[0]/f"draw_{key[1]}"/f"seed{key[2]}.npz"
        assert digest(path)==row["trace_sha256"]
        assert float(row["action_max_error"])<=1e-5
        assert float(row["max_qpos_error"])<3e-6 and float(row["max_qvel_error"])<1e-4
        with np.load(path,allow_pickle=False) as z:
            assert all(np.isfinite(z[name]).all() for name in z.files)
            assert int(z["active"].sum())==int(row["native_steps"])
            assert bool(z["success30"][-1])==(keyed[key]["success30"]=="True")
            physical+=int(z["active"].sum())
            if key[0]!="deterministic" and keyed[("deterministic",0,key[2])]["success30"]=="True" and not bool(z["success30"][-1]):
                active=z["active"].astype(bool)
                released=active&(z["audit"][:,10]>0)
                valid=released&(z["audit"][:,11]>0)
                death=active&~z["alive"].astype(bool)
                lost_details.append(dict(arm=key[0],draw=key[1],seed=key[2],
                    died_before_release=bool(death.any() and not released.any()),
                    reached_release=bool(released.any()),ever_valid=bool(valid.any()),
                    first_valid_sec=float(z["audit"][valid,12][0]) if valid.any() else None,
                    terminal_sec=float(z["audit"][death,12][0]) if death.any() else None,
                    best_hold_sec=float(z["best_hold"].max()),
                    released_depth_over_3mm=bool(np.any(z["audit"][released,6]>.003))))
    assert physical==summary["physical_interactions"]==summary["native_validation_interactions"]
    seeds=summary["seeds"]
    rows=[]
    for seed in seeds:
        row=dict(seed=seed)
        for arm,n in (("deterministic",1),("independent",2),("hold4",2)):
            row[arm]=float(np.mean([keyed[(arm,draw,seed)]["success30"]=="True" for draw in range(n)]))
        rows.append(row)
    rng=np.random.default_rng(270042)
    indices=rng.integers(0,len(rows),size=(10000,len(rows)))
    comparisons=[]
    for arm,control in (("hold4","independent"),("independent","deterministic"),("hold4","deterministic")):
        delta=np.array([r[arm]-r[control] for r in rows])
        ci=np.quantile(delta[indices].mean(axis=1),[.025,.975])
        comparisons.append(dict(method=arm,control=control,delta_pp=float(delta.mean()*100),
            low_pp=float(ci[0]*100),high_pp=float(ci[1]*100),
            seeds_improved=int((delta>0).sum()),seeds_worse=int((delta<0).sum())))
    out=folder/"report"
    out.mkdir(exist_ok=True)
    write_csv(out/"per_seed.csv",rows)
    write_csv(out/"paired_intervals.csv",comparisons)
    write_csv(out/"lost_success_timing.csv",lost_details)
    failure_groups=[]
    for arm in ("independent","hold4"):
        selected=[r for r in lost_details if r["arm"]==arm]
        failure_groups.append(dict(arm=arm,lost=len(selected),
            died_before_release=sum(r["died_before_release"] for r in selected),
            ever_valid=sum(r["ever_valid"] for r in selected),
            held_at_least_10s=sum(r["best_hold_sec"]>=10 for r in selected),
            released_depth_over_3mm=sum(r["released_depth_over_3mm"] for r in selected),
            survived_full_episode=sum(r["terminal_sec"] is None for r in selected)))
    record=dict(status="verified",comparisons=comparisons,summary_sha256=digest(folder/"summary.json"),
        episodes_sha256=digest(folder/"episodes.csv"),audits_sha256=digest(folder/"audits.csv"),
        source_sha256=digest(Path(__file__)),physical_interactions=physical,native_validation_interactions=physical,
        uncertainty="10000 paired percentile bootstrap draws over 32 initial-state clusters; retain both noise draws in each cluster",
        scope="Frozen-policy diagnostic, not learned-policy generalization or independent PPO training replicates",
        lost_success_timing=failure_groups)
    (out/"evidence.json").write_text(json.dumps(record,indent=2)+"\n")
    lines=["# D42按初态配对统计", "",record["uncertainty"], "",
        "| 比较 | 成功率差pp | 95%初态簇bootstrap区间pp | 改善初态/退化初态 |",
        "|---|---:|---:|---:|"]
    for r in comparisons:
        lines.append(f"| {r['method']} - {r['control']} | {r['delta_pp']:+.2f} | "
                     f"[{r['low_pp']:+.2f}, {r['high_pp']:+.2f}] | {r['seeds_improved']}/{r['seeds_worse']} |")
    lines += ["", "两个噪声重复随同一初态一起抽样，不当64个独立初态，也不是多个PPO训练种子。",
        "全部160个文件哈希、成功标签、动作/物理审计容差和交互成本已核对。",
        "区间描述本批诊断初态不确定性，样本小且只用一份BC；不能据此宣称正式训练有效。",
        "", "## 原本成功、加噪声后失败的回合", "",
        "| 组 | 丢失次数 | 撤绳前死亡 | 曾合格接触 | 曾连续合格至少10秒 | 释放后穿透超3mm | 存活完整40秒 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in failure_groups:
        lines.append(f"| {r['arm']} | {r['lost']} | {r['died_before_release']} | {r['ever_valid']} | "
                     f"{r['held_at_least_10s']} | {r['released_depth_over_3mm']} | {r['survived_full_episode']} |")
    lines += ["", "各列可重叠，是失败时序和接触描述，不是互斥根因。撤绳后才失败也不能证明扰动的影响只在撤绳后开始。",
        "逐例首次合格/死亡时刻、保持时长和穿透标志见lost_success_timing.csv；不存在的时刻留空，不填0。"]
    (out/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(record),flush=True)


if __name__=="__main__":
    main()
