"""Frozen BC: independent versus block-held noise at equal marginal std.

This diagnostic performs no PPO updates and does not claim to implement gSDE.
Sampling keys are explicit so every saved action and native trajectory is audited.
"""

import argparse
import json
from pathlib import Path
import pickle
import shutil
import sys

import jax
import jax.numpy as jp
import numpy as np

from analyze_s09_bc_confirmation import write_csv
from audit_s09_wide_actor import matched_actor
from probe_s09_feedback_perturbation import native_audit
from retarget_s09_reference import digest
from run_s09_bc_conditioned import ModeConditionedGrasp
import run_s09_job as runner

ROOT=Path(__file__).resolve().parents[1]


def noise_key(seed,draw):
    return jax.random.fold_in(jax.random.PRNGKey(seed),270000+draw)


def noise_at(key,index,repeat):
    return jax.random.normal(jax.random.fold_in(key,index//repeat),(17,))


def make_evaluator(env,net,steps):
    @jax.jit
    def evaluate(params,seeds,draw,repeat,std):
        def episode(seed):
            initial=env.reset(jax.random.PRNGKey(seed))
            key=noise_key(seed,draw)

            def step(carry,index):
                state,alive=carry
                logits=net.policy_network.apply(params[0],params[1],state.obs)
                mean=net.parametric_action_distribution.create_dist(logits).loc
                action=net.parametric_action_distribution.postprocess(mean+std*noise_at(key,index,repeat))
                nxt=jax.lax.cond(alive,lambda s:env.step(s,action),lambda s:s,state)
                next_alive=alive&(nxt.done==0)
                trace=dict(obs=state.obs["state"],action=action,active=alive,alive=next_alive,
                    qpos=nxt.data.qpos,qvel=nxt.data.qvel,ctrl=nxt.data.ctrl,audit=nxt.data.audit,
                    hold=jp.where(next_alive,nxt.metrics["hold_sec"],0),best_hold=nxt.metrics["best_hold_sec"],
                    success30=nxt.metrics["success30"]*next_alive,success15=nxt.metrics["success15"]*next_alive,
                    initial_penetration=nxt.info["initial_penetration"],reference_index=nxt.info["reference_index"],
                    requested_residual=nxt.info["requested_residual"],desired_ctrl=nxt.info["desired_ctrl"],
                    reward=jp.where(alive,nxt.reward,0))
                return (nxt,next_alive),trace

            _,trace=jax.lax.scan(step,(initial,jp.bool_(True)),jp.arange(steps))
            return initial,trace
        return jax.vmap(episode)(seeds)
    return evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    assert jax.default_backend()=="cpu"
    jax.config.update("jax_default_matmul_precision","highest")
    source=ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/I27_phase_s13102"
    checkpoint=source/"step_000000000/params.pkl"
    cfg=json.loads((source/"run_config.json").read_text())
    model=ROOT/"artifacts/orca_benchmark_suite/S09/inputs/model.xml"
    assert digest(model)==cfg["model_sha256"] and digest(model.parent/"reference.npz")==cfg["reference_sha256"]
    env=ModeConditionedGrasp(model,"pure")
    net,_=runner.initialize_matched(env,cfg["seed"],cfg["std"])
    params=jax.tree_util.tree_map(jp.asarray,pickle.loads(checkpoint.read_bytes()))
    count,batch,steps,draws=(2,2,80,1) if args.smoke else (32,8,800,2)
    seeds=np.arange(250000,250000+count)
    evaluate=make_evaluator(env,net,steps)
    actor=matched_actor(net.policy_network,batch)
    infer=jax.jit(lambda observations:actor.apply(params[0],params[1],dict(state=observations))[...,:17])

    @jax.jit
    def replay_actions(observations,ids,draw,repeat,std):
        mean=infer(observations)
        noise=jax.vmap(lambda seed:jax.vmap(lambda index:noise_at(noise_key(seed,draw),index,repeat))(
            jp.arange(steps)))(ids)
        return net.parametric_action_distribution.postprocess(mean+std*noise)

    args.output.mkdir(parents=True,exist_ok=False)
    sources=args.output/"sources"
    sources.mkdir()
    hashes={}
    for module in list(sys.modules.values()):
        filename=getattr(module,"__file__",None)
        if filename and Path(filename).resolve().parent==ROOT/"scripts" and filename.endswith(".py"):
            path=Path(filename).resolve()
            shutil.copyfile(path,sources/path.name)
            hashes[path.name]=digest(sources/path.name)
    arms=[("deterministic",1,0.,1),("independent",1,.03,draws),("hold4",4,.03,draws)]
    protocol=dict(status="running",checkpoint_sha256=digest(checkpoint),model_sha256=digest(model),
        reference_sha256=digest(model.parent/"reference.npz"),sources=hashes,seeds=seeds.tolist(),steps=steps,
        batch=batch,arms=arms,policy_updates=0,smoke=args.smoke,backend=jax.default_backend(),
        scope="New diagnostic/training initial states, not held-out success evaluation",
        noise="Independent joints; repeat a standard-normal vector for n steps before tanh; std .03 per joint",
        caveat="Constant-feature block noise, not full gSDE; no PPO likelihood or learning claims")
    (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    rows,audits,initials=[],[],{}
    for label,repeat,std,n_draws in arms:
        for draw in range(n_draws):
            for start in range(0,count,batch):
                ids=seeds[start:start+batch]
                initial,trace=jax.device_get(evaluate(params,jp.asarray(ids,jp.int32),jp.int32(draw),jp.int32(repeat),jp.float32(std)))
                assert all(np.isfinite(v).all() for v in trace.values())
                predicted=np.asarray(replay_actions(jp.asarray(trace["obs"]),jp.asarray(ids,jp.int32),
                    jp.int32(draw),jp.int32(repeat),jp.float32(std)))
                active=trace["active"].astype(bool)
                np.testing.assert_allclose(predicted[active],trace["action"][active],atol=1e-5,rtol=0)
                _,episodes=runner.summarize(trace,ids)
                folder=args.output/label/f"draw_{draw}"
                folder.mkdir(parents=True,exist_ok=True)
                for i,(seed,row) in enumerate(zip(ids,episodes)):
                    state=jax.tree_util.tree_map(lambda x:x[i],initial)
                    single=jax.tree_util.tree_map(lambda x:x[i],trace)
                    if int(seed) not in initials:
                        initials[int(seed)]=np.asarray(state.data.qpos)
                    np.testing.assert_array_equal(initials[int(seed)],state.data.qpos)
                    audit=native_audit(model,state,single)
                    pair=single["active"][1:]&single["active"][:-1]
                    assert pair.any(),"Need two active frames for continuity diagnostics"
                    row.update(arm=label,draw=draw,repeat=repeat,std=std,
                        normalized_action_step_rms=float(np.sqrt(np.mean(np.diff(single["action"],axis=0)[pair]**2))),
                        applied_ctrl_step_rms_rad=float(np.sqrt(np.mean(np.diff(single["ctrl"],axis=0)[pair]**2))))
                    path=folder/f"seed{seed}.npz"
                    np.savez_compressed(path,**single,initial_qpos=np.asarray(state.data.qpos))
                    audits.append(dict(arm=label,draw=draw,seed=int(seed),**audit,trace_sha256=digest(path),
                        action_max_error=float(np.max(np.abs(predicted[i][active[i]]-single["action"][active[i]]),initial=0))))
                    rows.append(row)
                print(json.dumps(dict(arm=label,draw=draw,completed=start+batch)),flush=True)
    write_csv(args.output/"episodes.csv",rows)
    write_csv(args.output/"audits.csv",audits)
    overview=[]
    for label,_,_,_ in arms:
        selected=[r for r in rows if r["arm"]==label]
        overview.append(dict(arm=label,n=len(selected),successes=sum(r["success30"] for r in selected),
            success_rate=float(np.mean([r["success30"] for r in selected])),
            hold_sec=float(np.mean([r["hold_sec"] for r in selected])),
            normalized_action_step_rms=float(np.mean([r["normalized_action_step_rms"] for r in selected])),
            applied_ctrl_step_rms_rad=float(np.mean([r["applied_ctrl_step_rms_rad"] for r in selected])),
            max_depth_mm=max(r["max_released_penetration_m"] for r in selected)*1000,
            peak_force_n=max(r["peak_force_n"] for r in selected)))
    independent={(r["seed"],r["draw"]):r for r in rows if r["arm"]=="independent"}
    held={(r["seed"],r["draw"]):r for r in rows if r["arm"]=="hold4"}
    paired=dict(gained=sum(held[k]["success30"] and not independent[k]["success30"] for k in held),
                lost=sum(independent[k]["success30"] and not held[k]["success30"] for k in held))
    write_csv(args.output/"overview.csv",overview)
    summary=dict(protocol,status="completed_diagnostic_only",rows=overview,hold4_vs_independent=paired,
        physical_interactions=sum(r["native_steps"] for r in audits),
        native_validation_interactions=sum(r["native_steps"] for r in audits),
        max_action_error=max(r["action_max_error"] for r in audits),
        checkpoint_unchanged=digest(checkpoint)==protocol["checkpoint_sha256"])
    assert summary["checkpoint_unchanged"]
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["# BC探索噪声的时间结构：零更新诊断", "",
        "同一I27均值网络，关节间仍独立，只将噪声从每步重采样改为每4步重采样；tanh前各关节边际标准差均0.03。",
        "确定性基线每初态一次；两种噪声每初态两次独立随机抽样（smoke仅一次）。不是独立的64个物理初态。", "",
        "| 方法 | 成功/次数 | 平均合格秒 | 动作步间RMS | 实际控制步间RMS rad | 最大穿透mm | 峰值N |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in overview:
        lines.append(f"| {r['arm']} | {r['successes']}/{r['n']} | {r['hold_sec']:.3f} | "
            f"{r['normalized_action_step_rms']:.6f} | {r['applied_ctrl_step_rms_rad']:.6f} | {r['max_depth_mm']:.3f} | {r['peak_force_n']:.3f} |")
    lines += ["",f"hold4对逐步独立噪声：新增{paired['gained']}、丢失{paired['lost']}。",
        "全部保存轨迹已重算抽样动作并独立原生回放；额外仿真和验证成本在summary.json单列。",
        "更平滑不自动意味着抓得更稳。零更新结果不等于PPO训练收益，也不是完整gSDE实现。",
        "4秒smoke不能形成30秒成功证据；正式40秒诊断与未见保留测试分开。"]
    (args.output/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
