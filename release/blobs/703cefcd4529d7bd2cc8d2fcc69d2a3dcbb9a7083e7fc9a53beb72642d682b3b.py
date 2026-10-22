"""Matched S10 reference collection, BC, PPO and execution-residual training."""

import argparse
import copy
import csv
import functools
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Aero-Hand/sim_rl/mujoco_playground"))

from brax.training.agents.ppo import networks, train
import jax
import jax.numpy as jp
import numpy as np
from mujoco_playground._src import wrapper

from orca_s10_env import OrcaS10Grasp
from s10_mobile_base import OUT, INPUT
from retarget_s09_reference import digest
from run_s09_job import summarize
from train_orca_matched import initialize_matched, fit_matched_bc
from train_orca_comparison import network_factory


def evaluator(env, net, reference=False):
    inference = networks.make_inference_fn(net)

    @jax.jit
    def evaluate(params, seeds):
        policy = inference(params, deterministic=True)

        def episode(key):
            state = env.reset(key)
            initial = state.data.qpos

            def step(carry, _):
                state, alive, key = carry
                key, subkey = jax.random.split(key)
                action = (env.ctrl_to_action(env.reference_base(state.info)) if reference else
                          policy(state.obs, subkey)[0])
                nxt = jax.lax.cond(alive, lambda s:env.step(s, action), lambda s:s, state)
                next_alive = alive & (nxt.done==0)
                trace = dict(obs=state.obs["state"], action=action, active=alive, alive=next_alive,
                    qpos=nxt.data.qpos, qvel=nxt.data.qvel, ctrl=nxt.data.ctrl, mocap=nxt.data.mocap_pos,
                    mocap_quat=nxt.data.mocap_quat, audit=nxt.data.audit,
                    hold=jp.where(next_alive, nxt.metrics["hold_sec"], 0),
                    best_hold=nxt.metrics["best_hold_sec"], success30=nxt.metrics["success30"]*next_alive,
                    success15=nxt.metrics["success15"]*next_alive, initial_penetration=nxt.info["initial_penetration"],
                    reference_index=nxt.info["reference_index"], video_base=nxt.info["video_base"],
                    requested_residual=nxt.info["requested_residual"], effective_residual=nxt.info["effective_residual"],
                    desired_ctrl=nxt.info["desired_ctrl"], reward=jp.where(alive, nxt.reward, 0))
                return (nxt, next_alive, key), trace

            return initial, jax.lax.scan(step, (state, jp.bool_(True), jax.random.fold_in(key,987654)),
                                         None, length=840)[1]
        return jax.vmap(episode)(jax.vmap(jax.random.PRNGKey)(seeds))
    return evaluate


def bc_arrays(path):
    with np.load(path) as z:
        obs, action, active = z["obs"], z["action"], z["active"].astype(bool)
    if len(obs)<4 or obs.shape[-1]!=123 or action.shape[-1]!=17:
        raise ValueError("At least four successful matched S10 demonstrations are required")
    train_mask = np.arange(len(obs))[:, None] < max(2, len(obs)*3//4)
    early = np.arange(obs.shape[1])[None, :] < 100
    masks = (active&train_mask&early, active&train_mask&~early, active&~train_mask)
    return tuple(jp.asarray(value[mask]) for mask in masks for value in (obs, action))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=("reference", "collect", "bc", "guided", "ppo", "residual"), required=True)
    p.add_argument("--selection", type=Path, default=OUT/"inputs/selection_v2.json")
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--evals", type=int, default=5)
    p.add_argument("--eval-count", type=int, default=16)
    p.add_argument("--eval-start", type=int, default=211000)
    p.add_argument("--eval-batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=22101)
    p.add_argument("--std", type=float, default=.1)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--bc-updates", type=int, default=20000)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()
    if a.steps and a.method in ("reference", "collect", "bc"):
        p.error("This method cannot update PPO")
    if a.eval_count % a.eval_batch:
        p.error("eval-count must divide by eval-batch")
    a.output.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_pmap_shmap_merge", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", str(OUT/"jax_cache"))
    if not a.allow_cpu:
        assert jax.default_backend()=="gpu", "GPU required; use host execution"
    selection = json.loads(a.selection.read_text())
    model = OUT/"inputs/model.xml"
    assert selection["model_sha256"]==digest(model)
    assert selection["reference_sha256"]==digest(INPUT/"reference.npz")
    env = OrcaS10Grasp(model, selection, "residual" if a.method=="residual" else "pure")
    net, params = initialize_matched(env, a.seed, a.std)
    meta = {k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items()}
    meta.update(model_sha256=digest(model), selection_sha256=digest(a.selection),
                reference_sha256=selection["reference_sha256"], physics_backend=env.physics_backend,
                actor_dim=123, critic_dim=140, env_config=env._config.to_dict(), backend=jax.default_backend())
    if a.checkpoint:
        folder = a.checkpoint.parent
        if not (folder/"run_config.json").exists():
            folder = folder.parent
        previous = json.loads((folder/"run_config.json").read_text())
        for key in ("model_sha256", "selection_sha256", "reference_sha256", "physics_backend"):
            if previous[key]!=meta[key]:
                raise ValueError("Checkpoint mismatch: "+key)
        if previous["method"]!=a.method and not (previous["method"]=="bc" and a.method=="guided"):
            raise ValueError("Incompatible action interpretation")
        params = pickle.loads(a.checkpoint.read_bytes())
        meta["checkpoint_sha256"] = digest(a.checkpoint)
    elif a.method in ("bc", "guided"):
        if a.dataset is None:
            p.error("BC needs a matched dataset")
        manifest = json.loads((a.dataset.parent/"dataset_manifest.json").read_text())
        for key in ("model_sha256", "selection_sha256", "physics_backend"):
            assert manifest[key]==meta[key], key
        assert manifest["dataset_sha256"]==digest(a.dataset)
        arrays = bc_arrays(a.dataset)
        # A single template depends on phase. PPO subsequently trains all inputs.
        mask = np.zeros(123, bool); mask[[93,110,111,112]] = True
        params = copy.deepcopy(params)
        first = params[1]["params"]["hidden_0"]
        first["kernel"] = first["kernel"] * jp.asarray(mask)[:,None]
        arrays = tuple(v*jp.asarray(mask) if i%2==0 else v for i,v in enumerate(arrays))
        params, losses = fit_matched_bc(net, params, arrays, a.bc_updates, a.seed)
        (a.output/"bc_loss.json").write_text(json.dumps(losses, indent=2)+"\n")
        (a.output/"bc_params.pkl").write_bytes(pickle.dumps(jax.device_get(params)))
        meta["dataset_sha256"] = digest(a.dataset)
        meta["bc_input_columns"] = np.flatnonzero(mask).tolist()
    sources = a.output/"sources"; sources.mkdir()
    meta["sources"] = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().parent==ROOT/"scripts" and filename.endswith(".py"):
            source = Path(filename).resolve()
            shutil.copyfile(source, sources/source.name)
            meta["sources"][source.name] = digest(source)
    (a.output/"run_config.json").write_text(json.dumps(meta, indent=2)+"\n")
    evaluate = evaluator(env, net, a.method in ("reference", "collect"))
    seeds = np.arange(a.eval_start, a.eval_start+a.eval_count)
    rows = []; started = time.monotonic()

    def callback(step, _policy, current):
        batches = [jax.device_get(evaluate(current, jp.asarray(seeds[i:i+a.eval_batch], jp.int32)))
                   for i in range(0,len(seeds),a.eval_batch)]
        initial, trace = jax.tree_util.tree_map(lambda *v:np.concatenate(v), *batches)
        assert all(np.isfinite(v).all() for v in trace.values())
        row, episodes = summarize(trace, seeds)
        assert row["released_rope_force_n"]==0
        row.update(step=int(step), elapsed_sec=time.monotonic()-started)
        rows.append(row)
        folder = a.output/f"step_{int(step):09d}"; folder.mkdir(exist_ok=False)
        (folder/"episodes.json").write_text(json.dumps(episodes, indent=2)+"\n")
        np.savez_compressed(folder/"initial_states.npz", qpos=initial, seeds=seeds)
        indices = list(range(min(2,len(seeds))))
        for success in (True,False):
            indices += [i for i,e in enumerate(episodes) if e["success30"]==success and i not in indices][:1]
        np.savez_compressed(folder/"preview_trajectories.npz", seeds=seeds[indices],
                            **{k:v[indices] for k,v in trace.items()})
        if a.method not in ("reference", "collect"):
            canonical = jax.tree_util.tree_map(np.asarray,jax.device_get(current))
            (folder/"params.pkl").write_bytes(pickle.dumps(canonical))
            (a.output/"params.pkl").write_bytes(pickle.dumps(canonical))
        with (a.output/"metrics.csv").open("w") as f:
            writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
        print(json.dumps(row), flush=True)
        if a.method=="collect":
            selected = np.array([e["success30"] for e in episodes])
            if selected.sum()<4:
                raise ValueError("Not enough successful reference demonstrations")
            path = a.output/"trajectories.npz"
            np.savez_compressed(path, **{k:trace[k][selected] for k in ("obs","action","active")})
            manifest = {k:meta[k] for k in ("model_sha256","selection_sha256","physics_backend")}
            manifest.update(dataset_sha256=digest(path), seeds=seeds[selected].tolist(),
                            attempted_seeds=seeds.tolist(), actor_dim=123)
            (a.output/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")

    def progress(step, metrics):
        with (a.output/"ppo_metrics.jsonl").open("a") as f:
            f.write(json.dumps(dict(step=int(step), **{k:float(v) for k,v in metrics.items()}))+"\n")

    if a.steps:
        train.train(environment=env,num_timesteps=a.steps,num_envs=32,num_eval_envs=a.eval_batch,
            episode_length=840,batch_size=32,num_minibatches=4,unroll_length=20,num_updates_per_batch=4,
            num_resets_per_eval=0,learning_rate=a.learning_rate,entropy_cost=.0001,discounting=.99,
            normalize_observations=False,reward_scaling=1,network_factory=network_factory,
            wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training,full_reset=True),
            seed=a.seed,num_evals=a.evals,run_evals=False,restore_params=params,restore_value_fn=True,
            policy_params_fn=callback,progress_fn=progress)
    else:
        callback(0,None,params)
    summary = dict(status="completed", final_step=rows[-1]["step"], rows=rows)
    for key in ("success30","hold_sec"):
        best=max(rows,key=lambda r:r[key])
        summary[key]=dict(first=rows[0][key],last=rows[-1][key],max=best[key],best_step=best["step"])
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("DONE",json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
