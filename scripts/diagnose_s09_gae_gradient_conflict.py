"""Compare critic gradients from matched-frame MC and Brax GAE targets."""

import argparse
import json
from pathlib import Path
import pickle
import shutil

from brax.training.agents.ppo import losses
import jax
import jax.numpy as jp
from jax.flatten_util import ravel_pytree
import numpy as np

from analyze_s09_bc_confirmation import write_csv
from diagnose_s09_critic_gradient_conflict import cosine_matrix
from retarget_s09_reference import digest
import run_s09_learned_base_residual as learned
from run_s09_learned_residual_experts import residual_expert_factory


def segmented_targets(rewards, values, active, length=20, gamma=.99, lambda_=.95):
    rewards, values, active = np.asarray(rewards), np.asarray(values), np.asarray(active, bool)
    if rewards.ndim != 2 or rewards.shape != values.shape or rewards.shape != active.shape:
        raise ValueError("Expected matching episode/time arrays")
    if length < 1 or not 0 < gamma < 1 or not 0 <= lambda_ <= 1:
        raise ValueError("Invalid GAE parameters")
    if not np.isfinite(rewards).all() or not np.isfinite(values).all():
        raise ValueError("Nonfinite rollout")
    if np.any(active[:,1:] & ~active[:,:-1]):
        raise ValueError("Diagnostic episodes must not restart after termination")
    batch, frames = rewards.shape
    usable = (frames-1)//length*length
    if not usable:
        raise ValueError("Need at least one complete segment and its next observation")
    segments = usable//length

    def time_first(x):
        return jp.asarray(x.reshape(batch*segments,length).T)

    # The last recorded observation is a possible bootstrap, never an invented terminal.
    reward = time_first(rewards[:,:usable])
    value = time_first(values[:,:usable])
    terminal = time_first((~active[:,1:usable+1]).astype(np.float32))
    bootstrap = jp.asarray(values[:,np.arange(1,segments+1)*length].reshape(-1))
    target, _ = losses.compute_gae(jp.zeros_like(reward),terminal,reward,value,bootstrap,
                                    lambda_=lambda_,discount=gamma)
    return np.asarray(target.T).reshape(batch,usable)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text())
    source = read(args.diagnostic/"summary.json")
    assert source["status"] == "completed" and source["policy_updates"] == source["critic_updates"] == 0
    path = args.diagnostic/"critic_calibration.npz"
    assert digest(path) == source["data_sha256"]
    run = Path(source["source_run"])
    cfg, protocol = read(run/"run_config.json"),read(run/"learned_base_config.json")
    checkpoint = run/"step_000000000/params.pkl"
    assert digest(checkpoint) == source["checkpoint_sha256"]
    for name in ("run_s09_learned_base_residual.py","run_s09_learned_residual_experts.py"):
        assert digest(Path(__file__).parent/name) == cfg["sources"][name]
    base_path,model = Path(protocol["base_checkpoint"]),Path(cfg["input"])/"model.xml"
    assert digest(base_path) == source["base_sha256"] and digest(model) == source["model_sha256"]
    jax.config.update("jax_default_matmul_precision","highest")
    base,_,_ = learned.load_base(base_path,model)
    net = residual_expert_factory(base)(dict(state=(123,),privileged_state=(140,)),17)
    params = jax.tree_util.tree_map(jp.asarray,pickle.loads(checkpoint.read_bytes()))
    with np.load(path,allow_pickle=False) as z:
        obs,rewards,active,mc,mode = z["privileged_obs"],z["reward"],z["active"],z["returns"],z["mode"]
    predict = jax.jit(lambda x:net.value_network.apply(params[0],params[2],
        dict(state=x[...,:123],privileged_state=x)))
    prediction = np.asarray(predict(jp.asarray(obs)))
    gae = segmented_targets(rewards,prediction,active)
    usable = gae.shape[1]
    observations = dict(state=jp.asarray(obs[:,:usable,:123]),privileged_state=jp.asarray(obs[:,:usable]))

    def loss(value,target,mask):
        estimate = net.value_network.apply(params[0],value,observations)
        # Brax defaults: half squared error times vf_coefficient=0.5.
        return .25*jp.sum(jp.where(mask,(estimate-target)**2,0))/jp.sum(mask)

    calculate = jax.jit(jax.value_and_grad(loss))
    rows,pairs,arrays,gradient_sets = [],[],{},{}
    frames = np.arange(usable)[None,:]
    for name,target in (("mc",mc[:,:usable]),("gae20",gae)):
        group_rows,gradients = [],[]
        for phase,times in (("all",frames>=0),("acquisition",frames<60),
                            ("released_before_30s",(frames>=60)&(frames<600))):
            for group in range(-1,6):
                cases = np.ones(len(mode),bool) if group==-1 else mode==group
                mask = active[:,:usable]&cases[:,None]&times
                if not mask.any():
                    continue
                value,grad = calculate(params[2],jp.asarray(target),jp.asarray(mask))
                vector = np.asarray(ravel_pytree(grad)[0],dtype=np.float64)
                assert np.isfinite(vector).all()
                group_rows.append(dict(target=name,phase=phase,mode=group,n=int(mask.sum()),
                    episodes=int(cases.sum()),value_loss=float(value),gradient_norm=float(np.linalg.norm(vector))))
                gradients.append(vector)
        gradients = np.stack(gradients)
        cosine = cosine_matrix(gradients)
        for i,row in enumerate(group_rows):
            row["cosine_with_global"] = float(cosine[i,0]) if np.isfinite(cosine[i,0]) else None
            row["dot_with_global"] = float(gradients[i]@gradients[0])
            for j,b in enumerate(group_rows[:i]):
                pairs.append(dict(target=name,a_phase=b["phase"],a_mode=b["mode"],
                    b_phase=row["phase"],b_mode=row["mode"],cosine=float(cosine[j,i])))
        rows.extend(group_rows)
        arrays[name+"_gradients"],arrays[name+"_cosine"] = gradients,cosine
        gradient_sets[name] = group_rows
    comparisons = []
    for i,(a,b) in enumerate(zip(gradient_sets["mc"],gradient_sets["gae20"])):
        assert (a["phase"],a["mode"],a["n"]) == (b["phase"],b["mode"],b["n"])
        cross = cosine_matrix([arrays["mc_gradients"][i],arrays["gae20_gradients"][i]])[0,1]
        comparisons.append(dict(phase=a["phase"],mode=a["mode"],n=a["n"],
            mc_global_cosine=a["cosine_with_global"],gae_global_cosine=b["cosine_with_global"],
            mc_vs_gae_cosine=float(cross)))
    args.output.mkdir(parents=True,exist_ok=False)
    for name,data in (("groups.csv",rows),("pairs.csv",pairs),("target_comparison.csv",comparisons)):
        write_csv(args.output/name,data)
    np.savez_compressed(args.output/"gradients.npz",**arrays,gae_targets=gae)
    summary = dict(status="completed_diagnostic_only",groups=rows,comparison=comparisons,
        usable_frames_per_episode=usable,active_frames=int(active[:,:usable].sum()),
        omitted_active_frames=int(active[:,usable:].sum()),gamma=.99,gae_lambda=.95,unroll_length=20,
        value_loss_coefficient=.5,extra_environment_interactions=0,policy_updates=0,critic_updates=0,
        source_sha256=digest(Path(__file__)),data_sha256=digest(path),checkpoint_sha256=digest(checkpoint),
        brax_loss_source_sha256=digest(Path(losses.__file__)),gradients_sha256=digest(args.output/"gradients.npz"),
        limitations="Frozen policy diagnostic rollouts, not optimizer minibatches; no resets after termination; "
            "exclude final segment with missing next observation; local value-loss gradients, not Adam or policy gradients")
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    shutil.copyfile(Path(__file__),args.output/Path(__file__).name)
    lines = ["# 完整回报与PPO短片段价值目标的梯度对照", "",
        "同一初始策略、critic与D34随机回合；调用本机Brax compute_gae，20步、gamma0.99、lambda0.95。",
        f"每例使用前{usable}帧，最后一段缺少下一观测而不使用；两种目标的帧掩码完全相同。",
        "这不是重新采集的真实优化小批，也不包含Adam动量或动作梯度；结论仅限固定critic的局部value loss。", "",
        "| 阶段 | 朝向 | MC对全局余弦 | GAE对全局余弦 | 同组MC对GAE余弦 |",
        "|---|---:|---:|---:|---:|"]
    for r in comparisons:
        lines.append(f"| {r['phase']} | {r['mode']} | {r['mc_global_cosine']:.4f} | "
                     f"{r['gae_global_cosine']:.4f} | {r['mc_vs_gae_cosine']:.4f} |")
    (args.output/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
