"""Independently audit wrap traces and aggregate the frozen S08 comparison."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from analyze_s07_study import write_csv, wilson, verify_run_files, COLORS, LABEL
from prepare_cylinder_s02_source import digest
from run_s08_study import OUT, ROOT, SEEDS
from run_s07_study import BENCHMARKS, METHODS, write_json
from orca_suite_env import BENCHMARKS as BENCHMARK_SETTINGS

REPORT=OUT/"report"


def verify_trace(trace,episodes,config,packet,model,gain,bound):
    assert all(np.isfinite(v).all() for v in trace.values())
    loads=trace["wrap_loads"]
    pair=(loads[...,0]>.02)&(loads[...,1]>.02)&((loads[...,2:]>.02).sum(axis=-1)>=2)
    np.testing.assert_array_equal(pair,trace["pad_pair"].astype(bool))
    active=trace["active"].astype(bool)
    valid=pair & active & trace["alive"].astype(bool) & (trace["released"]>0)
    valid &= (trace["cube_z"]>trace["spawn_z"]-.015)&(trace["initial_penetration"]<=.001)
    valid &= (trace["substep_forbidden"]==0)&(trace["substep_penetration"]<=.003)
    count=np.zeros(len(loads),int)
    holds=[]
    for t in range(800):
        count=np.where(valid[:,t],count+1,0)
        holds.append(count*.05)
    hold=np.stack(holds,axis=1)
    np.testing.assert_allclose(hold,trace["hold"],atol=5e-6)
    errors=[]
    low,high=model.actuator_ctrlrange.T
    for i,e in enumerate(episodes[:len(loads)]):
        mask=active[i]; released=mask&(trace["released"][i]>0)
        depth=float(np.max(trace["substep_penetration"][i,released],initial=0))
        forbidden=bool(np.any(trace["substep_forbidden"][i,released]>0))
        success=bool(hold[i].max()>=30 and valid[i,-1] and not forbidden and depth<=.003)
        assert success==e["success30"]
        assert depth==e["penetration_m"] and forbidden==e["forbidden"]
        assert abs(hold[i].max()-e["pad_hold_sec"])<5e-6
        assert float(trace["initial_penetration"][i,0])==e["initial_penetration_m"]
        if config["method"]=="residual":
            request=bound*np.clip(gain*trace["action"][i],-1,1)
            np.testing.assert_allclose(request[mask],trace["requested_residual"][i,mask],atol=2e-7)
            desired=np.clip(trace["video_base"][i]+request,low,high)
            np.testing.assert_allclose(trace["effective_residual"][i,mask],
                trace["ctrl"][i,mask]-trace["video_base"][i,mask],atol=2e-7)
        else:
            desired=low+(np.clip(trace["action"][i],-1,1)+1)*.5*(high-low)
        previous=np.vstack([packet["initial_ctrl"],trace["ctrl"][i,:-1]])
        predicted=np.clip(desired,previous-.1,previous+.1)
        error=float(np.max(np.abs(predicted[mask]-trace["ctrl"][i,mask])))
        assert error<1e-6,error
        errors.append(error)
    return len(loads),max(errors)


def main():
    REPORT.mkdir(exist_ok=True)
    complete=json.loads((OUT/"evaluation_complete.json").read_text())
    assert complete["jobs"]==86 and complete["selection_sha256"]==digest(OUT/"selection.json")
    frozen=json.loads((OUT/"selection.json").read_text())
    selection=frozen["selected"]
    assert frozen["protocol_sha256"]==digest(OUT/"protocol.json")
    for item in selection.values():
        assert digest(item["checkpoint"])==item["sha256"]
    for path,checksum in json.loads((OUT/"inputs_manifest.json").read_text()).items():
        assert digest(ROOT/path)==checksum
    training=[]
    for method in METHODS:
        for seed in SEEDS:
            run=OUT/"train"/f"{method}_s{seed}"
            cfg=verify_run_files(run)
            s=json.loads((run/"summary.json").read_text())
            assert s["status"]=="completed" and cfg["steps"]==163840
            assert [r["step"] for r in s["rows"]]==list(range(0,163841,20480))
            assert cfg["seed"]==seed and cfg["method"]==method and cfg["learning_rate"]==1e-4
            updates=[json.loads(x) for x in (run/"ppo_metrics.jsonl").read_text().splitlines()]
            assert [x["step"] for x in updates]==list(range(20480,163841,20480))
            assert all(np.isfinite(v) for x in updates for v in x.values())
            chosen=selection[run.name]["development"]
            bc=json.loads((run/"bc_loss.json").read_text()) if method=="guided" else []
            if method=="guided":
                assert cfg["dataset_sha256"]==digest(OUT/"demonstrations/trajectories.npz")
                assert cfg["bc_updates"]==20000
                assert bc[-1]["step"]==20000
                assert all(np.isfinite(v) for x in bc for v in x.values())
            training.append(dict(method=method,train_seed=seed,first=s["success30"]["first"],
                last=s["success30"]["last"],max=s["success30"]["max"],best_step=s["success30"]["best_step"],
                selected_step=chosen["step"],selected_success=chosen["success30"],
                first_kl=updates[0]["training/kl_mean"],last_kl=updates[-1]["training/kl_mean"],
                max_kl=max(x["training/kl_mean"] for x in updates),
                bc_first_train_loss=bc[0]["train_loss"] if bc else None,
                bc_last_train_loss=bc[-1]["train_loss"] if bc else None,
                bc_last_validation_loss=bc[-1]["validation_loss"] if bc else None))
    rows,summaries,initials,residual_stats,following=[],[],{},[],[]
    failures=[]
    sensitivity=[]
    verified,max_error=0,0.
    paths=sorted(OUT.glob("eval/*/*/summary.json"))
    assert len(paths)==86
    for path in paths:
        run=path.parent; bench,name=run.parent.name,run.name
        cfg=verify_run_files(run)
        env=json.loads((run/"env_config.json").read_text())["s07"]
        assert frozen["timestamp"]<=(run/"run_config.json").stat().st_mtime
        condition=bench if bench in BENCHMARKS else "B0_nominal"
        assert env["benchmark"]==condition and env["settings"]==BENCHMARK_SETTINGS[condition]
        inputs=OUT/("inputs_"+name if bench=="A_reference" else "inputs")
        assert cfg["model"]==str(inputs/"model.xml") and cfg["reference"]==str(inputs/"reference.npz")
        assert env["residual_bound"]==.08
        expected_gain=float(name.rsplit("_gain",1)[1]) if bench=="A_gain" else 1.
        assert env["residual_gain"]==expected_gain
        selected_name=name.split("_gain")[0]
        if selected_name in selection:
            item=selection[selected_name]
            assert cfg["method"]==item["method"] and cfg["seed"]==item["train_seed"]
            assert cfg["checkpoint"]==item["checkpoint"] and cfg["checkpoint_sha256"]==item["sha256"]
        else:
            assert cfg["method"]=="reference" and cfg["checkpoint"] is None
        assert np.isclose(env["object_mass"],.07 if bench=="B4_mass" else .035)
        np.testing.assert_allclose(env["object_friction"],[1.35*(.6 if bench=="B3_friction" else 1),.025,.004])
        folder=run/"step_000000000"
        episodes=json.loads((folder/"episodes.json").read_text())
        assert len(episodes)==(64 if bench=="B0_nominal" or bench.startswith("A_") else 32)
        assert [e["seed"] for e in episodes]==list(range(87000,87000+len(episodes)))
        group=bench if bench in BENCHMARKS else "B0_nominal"
        with np.load(folder/"initial_states.npz") as z:
            np.testing.assert_array_equal(initials.setdefault(group,z["qpos"].copy()),z["qpos"])
            np.testing.assert_array_equal(z["seeds"],[e["seed"] for e in episodes])
        method=name.split("_s")[0] if "_s" in name else name
        seed=int(name.split("_s")[1].split("_")[0]) if "_s" in name else -1
        for e in episodes:
            rows.append(dict(benchmark=bench,run=name,method=method,train_seed=seed,**e))
        k=sum(e["success30"] for e in episodes); n=len(episodes)
        summaries.append(dict(benchmark=bench,run=name,method=method,train_seed=seed,successes=k,count=n,
            success_rate=k/n,ci_low=wilson(k,n)[0],ci_high=wilson(k,n)[1],
            mean_hold=float(np.mean([e["pad_hold_sec"] for e in episodes])),
            max_depth_mm=max(e["penetration_m"] for e in episodes)*1000,
            max_initial_depth_mm=max(e["initial_penetration_m"] for e in episodes)*1000,
            invalid_initial_count=sum(e["initial_penetration_m"]>.001 for e in episodes),
            excess_penetration_count=sum(e["penetration_m"]>.003 for e in episodes),
            insufficient_hold_count=sum(e["pad_hold_sec"]<30 for e in episodes),
            final_invalid_count=sum(not e["final_valid"] for e in episodes),
            max_force_n=max(e["peak_force_n"] for e in episodes),forbidden=sum(e["forbidden"] for e in episodes)))
        with np.load(cfg["reference"]) as z:
            packet=dict(z)
        model=mujoco.MjModel.from_xml_path(cfg["model"])
        with np.load(folder/"preview_trajectories.npz") as z:
            trace=dict(z)
        count,error=verify_trace(trace,episodes,cfg,packet,model,env["residual_gain"],env["residual_bound"])
        verified+=count; max_error=max(max_error,error)
        if method=="residual":
            request=trace["requested_residual"][trace["active"].astype(bool)]
            effective=trace["effective_residual"][trace["active"].astype(bool)]
            residual_stats.append(dict(benchmark=bench,run=name,gain=env["residual_gain"],saved_episodes=count,
                request_rms_rad=float(np.sqrt(np.mean(request**2))),request_max_rad=float(np.abs(request).max()),
                effective_rms_rad=float(np.sqrt(np.mean(effective**2))),effective_max_rad=float(np.abs(effective).max()),
                saturation_fraction=float(np.mean(np.abs(request)>=.99*env["residual_bound"]))))
        if bench=="B0_nominal":
            loads=trace["wrap_loads"]
            contact_valid=(loads[...,0]>.02)&(loads[...,1]>.02)&((loads[...,2:]>.02).sum(-1)>=2)
            contact_valid &= trace["active"].astype(bool)&trace["alive"].astype(bool)&(trace["released"]>0)
            contact_valid &= (trace["initial_penetration"]<=.001)&(trace["substep_forbidden"]==0)&(trace["substep_penetration"]<=.003)
            for threshold_mm in (10,15,20):
                valid=contact_valid&(trace["cube_z"]>trace["spawn_z"]-threshold_mm/1000)
                counter=np.zeros(count,int);best=counter.copy()
                for t in range(800):
                    counter=np.where(valid[:,t],counter+1,0);best=np.maximum(best,counter)
                passed=(best>=600)&valid[:,-1]&np.array([not e["forbidden"] and e["penetration_m"]<=.003 for e in episodes])
                if threshold_mm==15:
                    np.testing.assert_array_equal(passed,[e["success30"] for e in episodes])
                sensitivity.append(dict(run=name,method=method,train_seed=seed,height_threshold_mm=threshold_mm,
                    successes=int(passed.sum()),count=count,success_rate=float(passed.mean()),
                    no_drop_termination_count=int(trace["alive"][:,-1].sum())))
            times=(np.arange(800)+1)*.05
            base=np.stack([np.interp(times,packet["time"],packet["ctrl"][:,j]) for j in range(17)],axis=-1)
            for i,e in enumerate(episodes):
                last=int(np.flatnonzero(trace["active"][i])[-1])
                reasons=[]
                if not e["success30"]:
                    if e["initial_penetration_m"]>.001: reasons.append("invalid_initial")
                    if e["forbidden"]: reasons.append("forbidden_contact")
                    if e["penetration_m"]>.003: reasons.append("excess_penetration")
                    if not trace["alive"][i,-1]: reasons.append("drop")
                    if trace["cube_z"][i,last]<=trace["spawn_z"][i,last]-.015: reasons.append("height_loss")
                    if trace["wrap_loads"][i,last,0]<=.02: reasons.append("no_palm_load")
                    if trace["wrap_loads"][i,last,1]<=.02: reasons.append("no_thumb_load")
                    if (trace["wrap_loads"][i,last,2:]>.02).sum()<2: reasons.append("too_few_other_fingers")
                    if e["pad_hold_sec"]<30: reasons.append("insufficient_continuous_hold")
                failures.append(dict(run=name,method=method,seed=e["seed"],success30=e["success30"],
                    reasons="|".join(reasons) if reasons else "success",duration_sec=(last+1)*.05,
                    height_loss_mm=float(trace["spawn_z"][i,last]-trace["cube_z"][i,last])*1000))
                for phase,mask in (("all_active",trace["active"][i]),("valid_hold",trace["hold"][i]>0)):
                    if mask.any():
                        following.append(dict(run=name,method=method,seed=e["seed"],phase=phase,
                            frames=int(mask.sum()),success30=e["success30"],
                            target_rms_rad=float(np.sqrt(np.mean((trace["ctrl"][i,mask]-base[mask])**2))),
                            palm_min_load_n=float(np.mean(trace["wrap_loads"][i,mask,0])),
                            thumb_min_load_n=float(np.mean(trace["wrap_loads"][i,mask,1]))))
        print("AUDIT",bench,name,k,n,flush=True)
    assert len(rows)==3424 and verified==1604
    aggregate=[]
    for bench in BENCHMARKS:
        for method in ("ppo","bc","guided","reference","residual"):
            group=[x for x in summaries if x["benchmark"]==bench and x["method"]==method]
            rates=[x["success_rate"] for x in group]
            aggregate.append(dict(benchmark=bench,method=method,policies=len(group),
                success_mean=float(np.mean(rates)),success_sd=float(np.std(rates,ddof=1)) if len(rates)>1 else 0,
                successes=sum(x["successes"] for x in group),episodes=sum(x["count"] for x in group),
                mean_hold=float(np.mean([x["mean_hold"] for x in group])),
                invalid_initial_count=sum(x["invalid_initial_count"] for x in group),
                excess_penetration_count=sum(x["excess_penetration_count"] for x in group),
                insufficient_hold_count=sum(x["insufficient_hold_count"] for x in group),
                final_invalid_count=sum(x["final_invalid_count"] for x in group),
                max_depth_mm=max(x["max_depth_mm"] for x in group),max_force_n=max(x["max_force_n"] for x in group)))
    paired=[]
    for bench in BENCHMARKS:
        for seed in SEEDS:
            for left,right in ((f"residual_s{seed}","reference"),(f"guided_s{seed}",f"bc_s{seed}"),
                               (f"residual_s{seed}",f"ppo_s{seed}"),(f"guided_s{seed}",f"ppo_s{seed}")):
                a=[x for x in rows if x["benchmark"]==bench and x["run"]==left]
                b=[x for x in rows if x["benchmark"]==bench and x["run"]==right]
                assert [x["seed"] for x in a]==[x["seed"] for x in b]
                wins=sum(x["success30"] and not y["success30"] for x,y in zip(a,b))
                losses=sum(y["success30"] and not x["success30"] for x,y in zip(a,b))
                paired.append(dict(benchmark=bench,left=left,right=right,train_seed=seed,count=len(a),
                    wins=wins,losses=losses,ties=len(a)-wins-losses,net_percentage_points=100*(wins-losses)/len(a)))
    for name,data in (("episodes",rows),("policy_summary",summaries),("aggregate",aggregate),
                      ("paired_changes",paired),("training_diagnosis",training),
                      ("residual_stats",residual_stats),("reference_following",following)):
        write_csv(REPORT/(name+".csv"),data)
    write_csv(REPORT/"nominal_failure_modes.csv",failures)
    write_csv(REPORT/"height_threshold_sensitivity.csv",sensitivity)
    keys=("qpos","ctrl","hold","active")
    ref=dict(np.load(OUT/"eval/B0_nominal/reference/step_000000000/preview_trajectories.npz"))
    zero=[]
    for seed in SEEDS:
        trace=dict(np.load(OUT/f"eval/A_gain/residual_s{seed}_gain0/step_000000000/preview_trajectories.npz"))
        diffs={k:float(np.abs(trace[k].astype(float)-ref[k].astype(float)).max()) for k in keys}
        zero.append(dict(seed=seed,differences=diffs,exact=all(v==0 for v in diffs.values())))
    write_json(REPORT/"zero_gain_replay_checks.json",zero)
    write_json(REPORT/"verification.json",dict(status="verified",jobs=86,total_test_episodes=len(rows),
        recomputed_saved_episodes=verified,control_max_error_rad=max_error,
        paired_initial_states=True,source_snapshots_verified=True,selection_sha256=digest(OUT/"selection.json"),
        frozen_before_testing=True,evaluated_checkpoints_match_selection=True,intervention_settings_verified=True,
        zero_gain_exact=all(z["exact"] for z in zero),reference_origin="P03 geometry, not video"))
    plot(aggregate,summaries,following,paired)


def plot(aggregate,summaries,following,paired):
    # Reuse the common comparison figures, with a dedicated non-video following plot.
    import analyze_s07_study as common
    common.OUT,common.REPORT,common.SEEDS=OUT,REPORT,SEEDS
    common.plot(aggregate,summaries,[])
    (REPORT/"source_similarity.png").unlink()
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    methods=("ppo","bc","guided","reference","residual")
    for ax,phase in zip(axes,("all_active","valid_hold")):
        groups=[[r["target_rms_rad"] for r in following if r["phase"]==phase and r["method"]==m] for m in methods]
        ax.bar(range(5),[np.mean(g) if g else np.nan for g in groups],color=[COLORS[m] for m in methods])
        for j,group in enumerate(groups):
            if not group:
                ax.text(j,0,"N/A",ha="center",va="bottom")
        ax.set(xticks=range(5),xticklabels=["PPO","BC","BC+PPO","Reference","Residual"],
               ylabel="Target command RMS deviation (rad)",title=phase+" (not human-video similarity)")
    fig.savefig(REPORT/"reference_following.png",dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for ax,left,right,title in zip(axes,("residual","guided"),("reference","bc"),
                                  ("Residual vs reference","BC+PPO vs BC")):
        group=[r for r in paired if r["benchmark"]=="B0_nominal"
               and r["left"].startswith(left+"_s") and r["right"].startswith(right)]
        x=np.arange(len(group))
        ax.bar(x,[r["wins"] for r in group],color="#258b78",label="Rescued")
        ax.bar(x,[-r["losses"] for r in group],color="#b66a3c",label="Lost")
        ax.axhline(0,color="#555555",linewidth=.8)
        ax.set(xticks=x,xticklabels=[str(r["train_seed"]) for r in group],title=title,
               xlabel="Training seed",ylabel="Paired outcomes out of 64 initial states")
        ax.legend()
    fig.savefig(REPORT/"paired_outcomes.png",dpi=160); plt.close(fig)
    plot_wrap_example()


def plot_wrap_example():
    folder=OUT/f"eval/B0_nominal/residual_s{SEEDS[0]}/step_000000000"
    episodes=json.loads((folder/"episodes.json").read_text())
    with np.load(folder/"preview_trajectories.npz") as z:
        trace={k:z[k] for k in ("active","spawn_z","cube_z","wrap_loads")}
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout="constrained")
    for column,success in enumerate((True,False)):
        index=next((i for i,e in enumerate(episodes) if e["success30"]==success),None)
        if index is None:
            for ax in axes[:,column]:
                ax.set_axis_off()
            continue
        mask=trace["active"][index].astype(bool)
        t=((np.arange(800)+1)*.05)[mask]
        loss=(trace["spawn_z"][index,mask]-trace["cube_z"][index,mask])*1000
        ax=axes[0,column]
        ax.plot(t,loss,color="#4486b8")
        ax.axhline(15,color="#b66a3c",linestyle="--",label="15 mm limit after release")
        ax.set(ylabel="Object height loss (mm)",title=f"Seed {episodes[index]['seed']}: {'success' if success else 'failure'}\nFinal loss = {loss[-1]:.3f} mm")
        ax.legend(fontsize=8)
        loads=trace["wrap_loads"][index,mask]
        values=(loads[:,0],loads[:,1],np.partition(loads[:,2:],-2,axis=1)[:,-2])
        ax=axes[1,column]
        for y,label in zip(values,("Palm","Thumb","Second strongest other finger")):
            ax.plot(t,y,label=label)
        ax.axhline(.02,color="#b66a3c",linestyle="--",label="0.02 N threshold")
        ax.set_yscale("symlog",linthresh=.02)
        ax.set_ylim(bottom=0)
        ax.set(xlabel="Time (s)",ylabel="Minimum qualifying load per 50 ms (N)")
        ax.legend(fontsize=8)
        for ax in axes[:,column]:
            ax.axvline(3,color="#8b9198",linestyle=":")
            ax.set_xlim(0,40)
    fig.suptitle("First success/failure of residual training seed 8501; examples, not population means",fontsize=10)
    fig.savefig(REPORT/"wrap_contact_examples.png",dpi=160);plt.close(fig)


if __name__=="__main__":
    main()
