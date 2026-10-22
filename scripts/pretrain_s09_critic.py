"""Fit only the value network to complete on-policy BC returns before PPO.

All collection episodes, including failures, are retained. Splitting by episode
avoids putting neighboring states from one trajectory in train and validation.
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
import optax

import run_s09_ppo_improvement as improvement
from run_s09_bc_conditioned import ModeConditionedGrasp
from run_s09_bc_trust import with_std
from retarget_s09_reference import digest


ROOT=improvement.runner.ROOT
BASE=ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/critic_pretraining_20261018"


def returns_to_go(rewards,active,discount=.99):
    rewards,active=np.asarray(rewards),np.asarray(active,bool)
    if rewards.shape!=active.shape or rewards.ndim!=2 or not 0<discount<1:
        raise ValueError("Expected episode/time arrays and discount in (0,1)")
    result=np.zeros_like(rewards,dtype=np.float32)
    tail=np.zeros(rewards.shape[0],np.float64)
    for t in range(rewards.shape[1]-1,-1,-1):
        tail=np.where(active[:,t],rewards[:,t]+discount*tail,0.)
        result[:,t]=tail
    return result


def collector(env,net,deterministic=True):
    factory=improvement.runner.networks.make_inference_fn(net)

    @jax.jit
    def collect(params,seeds):
        policy=factory(params,deterministic=deterministic)

        def episode(key):
            state=env.reset(key)
            initial=state.data.qpos

            def step(carry,_):
                state,alive,key=carry
                key,subkey=jax.random.split(key)
                action=policy(state.obs,subkey)[0]
                nxt=jax.lax.cond(alive,lambda s:env.step(s,action),lambda s:s,state)
                trace=dict(privileged_obs=state.obs["privileged_state"],action=action,
                    reward=jp.where(alive,nxt.reward,0),active=alive)
                return (nxt,alive&(nxt.done==0),key),trace

            return initial,jax.lax.scan(step,(state,jp.bool_(True),jax.random.fold_in(key,987654)),None,length=800)[1]

        return jax.vmap(episode)(jax.vmap(jax.random.PRNGKey)(seeds))

    return collect


def fit_value(net,params,train_x,train_y,val_x,val_y,updates,seed):
    stats,actor,value=params
    optimizer=optax.adam(3e-4)
    optimizer_state=optimizer.init(value)
    train_x,train_y,val_x,val_y=map(jp.asarray,(train_x,train_y,val_x,val_y))

    def loss(value,x,y):
        prediction=net.value_network.apply(stats,value,dict(state=x[...,:123],privileged_state=x))
        return jp.mean((prediction-y)**2)

    @jax.jit
    def update(value,state,key):
        indices=jax.random.randint(key,(256,),0,len(train_x))
        _,gradient=jax.value_and_grad(loss)(value,train_x[indices],train_y[indices])
        delta,state=optimizer.update(gradient,state,value)
        return optax.apply_updates(value,delta),state

    evaluate=jax.jit(loss)
    rows=[]
    key=jax.random.PRNGKey(seed)
    for step in range(updates+1):
        if step%100==0 or step==updates:
            row=dict(update=step,train_rmse=float(jp.sqrt(evaluate(value,train_x,train_y))),
                     validation_rmse=float(jp.sqrt(evaluate(value,val_x,val_y))))
            rows.append(row)
            print(json.dumps(row),flush=True)
        if step<updates:
            key,subkey=jax.random.split(key)
            value,optimizer_state=update(value,optimizer_state,subkey)
    return (stats,actor,value),rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--episodes",type=int,default=128)
    parser.add_argument("--updates",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=16102)
    parser.add_argument("--collection-std",type=float,default=.003)
    parser.add_argument("--allow-cpu",action="store_true")
    args=parser.parse_args()
    if args.episodes<8 or args.episodes>128 or args.episodes%8 or args.updates<=0:
        parser.error("Use 8..128 episodes in batches of 8, and positive fitting updates")
    if not .001<args.collection_std<=1:
        parser.error("Collection std must match the intended PPO initialization and lie in (.001,1]")
    if not args.output.resolve().is_relative_to(BASE):
        parser.error("Keep value pretraining in its own provenance root")
    parent=args.checkpoint.parent.parent
    original_config=json.loads((parent/"run_config.json").read_text())
    if original_config["steps"]!=0 or original_config["method"]!="guided":
        parser.error("Start from an original BC-only checkpoint")
    if not (parent/"phase_curriculum.json").exists():
        parser.error("Use the repaired phase/mode BC origin")
    jax.config.update("jax_default_matmul_precision","highest")
    if not args.allow_cpu and jax.default_backend()!="gpu":
        parser.error("Use authorized GPU execution or explicitly allow CPU collection")
    inputs=Path(original_config["input"])
    env=ModeConditionedGrasp(inputs/"model.xml")
    net,_=improvement.runner.initialize_matched(env,args.seed,.03)
    params=pickle.loads(args.checkpoint.read_bytes())
    args.output.mkdir(parents=True,exist_ok=False)
    sources=args.output/"sources"
    sources.mkdir()
    hashes={}
    for module in list(sys.modules.values()):
        path=getattr(module,"__file__",None)
        if path and Path(path).resolve().parent==ROOT/"scripts" and path.endswith(".py"):
            path=Path(path).resolve()
            shutil.copyfile(path,sources/path.name)
            hashes[path.name]=digest(path)
    seeds=np.arange(140000,140000+args.episodes)
    collect=collector(env,net,deterministic=False)
    collection_params=with_std(params,args.collection_std)
    batches=[]
    for start in range(0,args.episodes,8):
        batches.append(jax.device_get(collect(collection_params,jp.asarray(seeds[start:start+8],jp.int32))))
        print(json.dumps(dict(collected_episodes=start+8)),flush=True)
    initial,trace=jax.tree_util.tree_map(lambda *xs:np.concatenate(xs),*batches)
    targets=returns_to_go(trace["reward"],trace["active"])
    assert np.isfinite(targets).all() and np.isfinite(trace["privileged_obs"]).all()
    split=args.episodes*3//4
    training=trace["active"].copy()
    training[split:]=False
    validation=trace["active"].copy()
    validation[:split]=False
    data_path=args.output/"critic_rollouts.npz"
    np.savez_compressed(data_path,**trace,returns=targets,seeds=seeds,initial_qpos=initial)
    fitted,rows=fit_value(net,params,trace["privileged_obs"][training],targets[training],
        trace["privileged_obs"][validation],targets[validation],args.updates,args.seed)
    for before,after in zip(jax.tree_util.tree_leaves(params[:2]),jax.tree_util.tree_leaves(fitted[:2])):
        np.testing.assert_array_equal(before,after)
    folder=args.output/"step_000000000"
    folder.mkdir()
    fitted=jax.tree_util.tree_map(np.asarray,jax.device_get(fitted))
    (folder/"params.pkl").write_bytes(pickle.dumps(fitted))
    config=dict(original_config,output=str(args.output.resolve()),checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=digest(args.checkpoint),backend=jax.default_backend(),steps=0,
        eval_count=0,eval_start=None,sources=hashes)
    (args.output/"run_config.json").write_text(json.dumps(config,indent=2)+"\n")
    metadata=dict(status="completed",source_checkpoint=str(args.checkpoint.resolve()),
        source_checkpoint_sha256=digest(args.checkpoint),actor_and_normalizer_exactly_unchanged=True,
        actor_action_law="Full target BC actor unchanged; no reference addition",discount=.99,
        collection_backend=jax.default_backend(),collection_policy="BC means with specified PPO exploration scale; failures retained",
        collection_std=args.collection_std,
        collection_seed_range=[int(seeds[0]),int(seeds[-1])],training_episodes=split,
        validation_episodes=args.episodes-split,actual_environment_interactions=int(trace["active"].sum()),
        ppo_interactions=0,fit_seed=args.seed,fit_updates=args.updates,fit_batch_size=256,fit_learning_rate=.0003,
        monte_carlo_target="Finite 40s episode, terminal future return zero, discounted R0 rewards",
        dataset_sha256=digest(data_path),checkpoint_sha256=digest(folder/"params.pkl"),fit_metrics=rows,
        limitations="Extra on-policy environment data, not free initialization; CPU/GPU reset streams differ; not a success evaluation")
    (args.output/"critic_pretraining.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata),flush=True)


if __name__ == "__main__":
    main()
