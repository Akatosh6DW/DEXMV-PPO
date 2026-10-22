"""Gated reconstruction study: frozen inputs, reward components and physical scoring."""

import argparse
import csv
import functools
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

from brax.training.agents.ppo import networks, train
import jax
import jax.numpy as jp
import numpy as np

from mujoco_playground._src import wrapper
from orca_guidance_recheck import REFERENCE, DATASET, score
from orca_refit_env import OrcaRefitGrasp, BASE_WEIGHTS
from train_orca_comparison import fit_bc, network_factory
from train_orca_guidance_recheck import curve_summary
from train_orca_residual import initialize_zero_mean

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_gates(model, gate_path):
    gate=json.loads(gate_path.read_text())
    native_path=Path(gate.get('validation_report',gate_path.parent.parent/'G07_validation/report.json'))
    proof=json.loads(native_path.read_text())
    geometry=json.loads((model.parent/'geometry_report.json').read_text())
    assert gate['pass_mjx'] and proof['geometry_and_native_pass']
    assert gate['model_sha256']==proof['model_sha256']==geometry['model_sha256']==digest(model)
    for path,expected in geometry['source_meshes'].items():
        assert digest(Path(path))==expected, f'Mesh changed: {path}'
    return dict(mjx=digest(gate_path),native=digest(native_path),
                geometry=digest(model.parent/'geometry_report.json'))


def evaluator(env,net,seeds):
    policy=networks.make_inference_fn(net)
    keys=jax.vmap(jax.random.PRNGKey)(jp.array(seeds,jp.int32))
    def evaluate(params):
        inference=policy(params,deterministic=True)
        def episode(key):
            state=env.reset(key)
            assert state.obs['state'].shape==(113,) and state.obs['privileged_state'].shape==(130,)
            key=jax.random.fold_in(key,987654)
            def step(carry,_):
                state,alive,key=carry
                key,sub=jax.random.split(key)
                action=inference(state.obs,sub)[0]
                nxt=env.step(state,action)
                next_alive=alive&(nxt.done==0)
                t=dict(obs=state.obs['state'],action=action,alive=next_alive,active=alive,
                       hold=jp.where(next_alive,nxt.metrics['hold_sec'],0),reward=jp.where(alive,nxt.reward,0),
                       qpos=nxt.data.qpos,qvel=nxt.data.qvel,ctrl=nxt.data.ctrl,mocap=nxt.data.mocap_pos,
                       forbidden=nxt.metrics['forbidden_contact'],contacts=nxt.metrics['contact_count'],
                       thumb=nxt.metrics['thumb_contact'],released=nxt.metrics['released'],
                       cube_z=nxt.metrics['cube_z'],spawn_z=nxt.info['cube_spawn_pos'][2],
                       cube_speed=nxt.metrics['cube_speed'],actuator_force_rms=nxt.metrics['actuator_force_rms'])
                for name in ('video_base','requested_residual','effective_residual','desired_ctrl'):
                    t[name]=nxt.info[name]
                for name in BASE_WEIGHTS:
                    t['component_'+name]=nxt.metrics['component_'+name]
                return (nxt,next_alive,key),t
            return jax.lax.scan(step,(state,jp.bool_(True),key),None,length=800)[1]
        return jax.vmap(episode)(keys)
    return jax.jit(evaluate)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=('pure','bc','residual'),required=True)
    p.add_argument('--weights',type=Path)
    p.add_argument('--seed',type=int,default=101)
    p.add_argument('--steps',type=int,default=327680)
    p.add_argument('--evals',type=int,default=5)
    p.add_argument('--eval-start',type=int,default=9200)
    p.add_argument('--eval-count',type=int,default=16)
    p.add_argument('--learning-rate',type=float,default=3e-5)
    p.add_argument('--speed-scale',type=float,default=.05)
    p.add_argument('--std',type=float)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--hic53-reference',type=Path,
                   help='Use the separately validated source-sized HIC task; never the old 36mm BC data')
    args=p.parse_args()
    if args.steps<0 or args.eval_count<1 or (args.steps>0 and args.evals<2):
        p.error('Invalid training/evaluation budget')
    reference=args.hic53_reference or REFERENCE
    if args.hic53_reference:
        from orca_hic53_env import OrcaHic53Grasp, check_hic53_gates
        if args.arm=='bc':
            p.error('HIC53 requires newly collected demonstrations; old 36mm BC data are not compatible')
        gates=check_hic53_gates(args.model,reference,args.gate)
    else:
        gates=check_gates(args.model,args.gate)
    if args.checkpoint and args.steps:
        p.error('Saved parameters are evaluation-only, not an optimizer continuation')
    args.output.mkdir(parents=True,exist_ok=False)
    jax.config.update('jax_default_matmul_precision','highest')
    jax.config.update('jax_compilation_cache_dir',str(ROOT/'artifacts/orca30/jax_cache'))
    started=time.monotonic()
    assert jax.default_backend()=='gpu', 'Use the GPU environment for this study'
    weights=json.loads(args.weights.read_text()) if args.weights else BASE_WEIGHTS
    env_type=OrcaHic53Grasp if args.hic53_reference else OrcaRefitGrasp
    env=env_type(args.model,args.arm,reference,weights,speed_scale=args.speed_scale)
    std=args.std if args.std is not None else (.03 if args.arm=='bc' else .7)
    net,params=initialize_zero_mean(args.seed,std)
    bc_sec=0.
    if args.checkpoint:
        params=pickle.loads(args.checkpoint.read_bytes())
    elif args.arm=='bc':
        before=time.monotonic()
        params,losses=fit_bc(net,params,DATASET,2000,args.seed)
        bc_sec=time.monotonic()-before
        (args.output/'bc_loss.json').write_text(json.dumps(losses))
    meta={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    meta.update(weights=weights,init_std=std,bc_sec=bc_sec,model_sha256=digest(args.model),gate_sha256=gates,
                reference_sha256=digest(reference),dataset_sha256=None if args.hic53_reference else digest(DATASET),
                task_name='hic53_source_matched_grasp' if args.hic53_reference else 'legacy36_collision_refit',
                checkpoint_sha256=digest(args.checkpoint) if args.checkpoint else None,
                backend=jax.default_backend(),command=sys.argv,source_sha256={},
                primary='unchanged 30s clean hold plus valid final frame',bc_updates=2000 if args.arm=='bc' and not args.checkpoint else 0,
                demo_loss_weight=0,observation_normalization=False)
    sources=args.output/'sources';sources.mkdir()
    for name in ('train_orca_refit.py','orca_refit_env.py','orca_residual_env.py','orca_benchmark_env.py',
                 'orca_residual_controls.py','orca_residual_metrics.py','train_orca_residual.py','train_orca_comparison.py',
                 'orca_guidance_recheck.py','train_orca_guidance_recheck.py',
                 'orca_hic53_env.py','prepare_hic_orca_object.py','orca_contact_forces.py'):
        path=ROOT/'scripts'/name
        (sources/name).write_bytes(path.read_bytes());meta['source_sha256'][name]=digest(path)
    (args.output/'run_config.json').write_text(json.dumps(meta,indent=2))
    (args.output/'env_config.json').write_text(env._config.to_json(indent=2))
    seeds=list(range(args.eval_start,args.eval_start+args.eval_count))
    evaluate=evaluator(env,net,seeds)
    rows=[]
    def callback(step,_policy,current):
        trace=jax.device_get(evaluate(current))
        assert all(np.isfinite(v).all() for v in trace.values())
        row,holds,success=score(trace,env)
        active=trace['active']
        for name in BASE_WEIGHTS:
            row['component_'+name]=float(np.mean(trace['component_'+name][active]))
        row['active_cube_speed']=float(np.mean(trace['cube_speed'][active]))
        row['active_force_rms']=float(np.mean(trace['actuator_force_rms'][active]))
        row.update(step=int(step),elapsed_sec=time.monotonic()-started)
        rows.append(row)
        with (args.output/'metrics.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
        directory=args.output/f'step_{int(step):09d}';directory.mkdir()
        (directory/'params.pkl').write_bytes(pickle.dumps(jax.device_get(current)))
        np.savez_compressed(directory/'trajectories.npz',**trace)
        (directory/'episodes.json').write_text(json.dumps(dict(seeds=seeds,hold_sec=holds.tolist(),success30=success.tolist())))
        print(json.dumps(row),flush=True)
    def progress(step,metrics):
        values={k:float(v) for k,v in metrics.items()}
        assert np.isfinite(list(values.values())).all()
        with (args.output/'ppo_metrics.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=int(step),**values))+'\n')
    if args.steps==0:
        callback(0,None,params)
    else:
        train.train(environment=env,num_timesteps=args.steps,num_envs=128,num_eval_envs=args.eval_count,
                    episode_length=800,batch_size=128,num_minibatches=4,unroll_length=20,
                    num_updates_per_batch=4,num_resets_per_eval=0,learning_rate=args.learning_rate,
                    entropy_cost=.001,discounting=.99,normalize_observations=False,reward_scaling=1,
                    network_factory=network_factory,
                    wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training,full_reset=True),
                    seed=args.seed,num_evals=args.evals,run_evals=False,restore_params=params,restore_value_fn=True,
                    policy_params_fn=callback,progress_fn=progress)
    result=curve_summary(rows)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2))
    print('DONE',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
