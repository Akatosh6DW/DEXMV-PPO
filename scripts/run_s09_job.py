"""Versioned S09 collection, matched PPO/BC training and held-out evaluation."""

import argparse
import csv
import functools
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"Aero-Hand/sim_rl/mujoco_playground"))

from brax.training.agents.ppo import networks, train
import jax
import jax.numpy as jp
import numpy as np
from mujoco_playground._src import wrapper

from orca_s09_env import OrcaS09Grasp
from retarget_s09_reference import digest
from train_orca_comparison import network_factory
from train_orca_matched import initialize_matched, dataset_arrays, fit_matched_bc


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
                nxt = jax.lax.cond(alive, lambda s: env.step(s, action), lambda s: s, state)
                next_alive = alive & (nxt.done==0)
                trace = dict(obs=state.obs["state"], action=action, active=alive, alive=next_alive,
                    qpos=nxt.data.qpos, qvel=nxt.data.qvel, ctrl=nxt.data.ctrl, mocap=nxt.data.mocap_pos,
                    audit=nxt.data.audit, hold=jp.where(next_alive, nxt.metrics["hold_sec"], 0),
                    best_hold=nxt.metrics["best_hold_sec"], success30=nxt.metrics["success30"]*next_alive,
                    success15=nxt.metrics["success15"]*next_alive, initial_penetration=nxt.info["initial_penetration"],
                    reference_index=nxt.info["reference_index"], video_base=nxt.info["video_base"],
                    requested_residual=nxt.info["requested_residual"], effective_residual=nxt.info["effective_residual"],
                    desired_ctrl=nxt.info["desired_ctrl"], reward=jp.where(alive, nxt.reward, 0))
                return (nxt, next_alive, key), trace

            return initial, jax.lax.scan(step, (state, jp.bool_(True), jax.random.fold_in(key,987654)), None, length=800)[1]

        return jax.vmap(episode)(jax.vmap(jax.random.PRNGKey)(seeds))
    return evaluate


def summarize(trace, seeds):
    episodes = []
    for i, seed in enumerate(seeds):
        active = trace["active"][i].astype(bool)
        audit = trace["audit"][i]
        released = active & (audit[:,10]>0)
        episodes.append(dict(seed=int(seed), success30=bool(trace["success30"][i,-1]),
            success15=bool(trace["success15"][i,-1]), hold_sec=float(trace["best_hold"][i].max()),
            final_valid=bool(trace["alive"][i,-1] and trace["hold"][i,-1]>0),
            duration_sec=float(active.sum()*.05), initial_penetration_m=float(trace["initial_penetration"][i,0]),
            max_released_penetration_m=float(np.max(audit[released,6], initial=0)),
            forbidden=bool(np.any(audit[released,5]>0)),
            released_rope_force_n=float(np.max(audit[released,8], initial=0)),
            peak_force_n=float(np.max(audit[active,7], initial=0)),
            residual_rms_rad=float(np.sqrt(np.mean(trace["requested_residual"][i,active]**2))),
            selected_reference=int(trace["reference_index"][i,0])))
    row = {key:float(np.mean([e[key] for e in episodes])) for key in ("success30","success15","hold_sec","residual_rms_rad")}
    row.update(max_penetration_m=max(e["max_released_penetration_m"] for e in episodes),
        max_initial_penetration_m=max(e["initial_penetration_m"] for e in episodes),
        forbidden_rate=float(np.mean([e["forbidden"] for e in episodes])),
        released_rope_force_n=max(e["released_rope_force_n"] for e in episodes),
        peak_force_n=max(e["peak_force_n"] for e in episodes))
    return row, episodes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=("ppo","guided","residual","reference","collect","bc"), required=True)
    p.add_argument("--steps", type=int, default=163840)
    p.add_argument("--evals", type=int, default=5)
    p.add_argument("--eval-count", type=int, default=8)
    p.add_argument("--eval-start", type=int, default=92000)
    p.add_argument("--eval-batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=9101)
    p.add_argument("--std", type=float, default=.3)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--approach-weight", type=float, default=2.)
    p.add_argument("--coverage-weight", type=float, default=2.)
    p.add_argument("--orientation", choices=("uniform","web","x","y","z"), default="uniform")
    p.add_argument("--dataset", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--bc-updates", type=int, default=4000)
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()
    if a.method in ("reference","collect","bc") and a.steps:
        p.error("Reference/collect/BC require --steps 0")
    if a.steps and a.orientation != "uniform":
        p.error("Do not replace the multi-orientation training task with a fixed test axis")
    if a.eval_count % a.eval_batch:
        p.error("eval-count must be divisible by eval-batch")
    a.output.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_pmap_shmap_merge", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", str(ROOT/"artifacts/orca30/jax_cache"))
    if not a.allow_cpu:
        assert jax.default_backend()=="gpu", "Use authorized host GPU execution"
    model, bank = a.input/"model.xml", a.input/"reference.npz"
    reference_mode = a.method in ("reference","collect")
    env = OrcaS09Grasp(model, "residual" if a.method=="residual" else "pure",
        bank if a.method=="residual" or reference_mode else None,
        approach_weight=a.approach_weight, coverage_weight=a.coverage_weight, orientation_mode=a.orientation)
    net, params = initialize_matched(env, a.seed, a.std)
    meta = {k:str(v.resolve()) if isinstance(v, Path) else v for k,v in vars(a).items()}
    meta.update(model_sha256=digest(model), reference_sha256=digest(bank),
        actor_dim=123, critic_dim=140, physics_backend=env.physics_backend, backend=jax.default_backend(),
        env_config=env._config.to_dict(), test_web_axis=env.web_axis.tolist())
    if a.checkpoint:
        folder = a.checkpoint.parent
        if not (folder/"run_config.json").exists():
            folder = folder.parent
        old = json.loads((folder/"run_config.json").read_text())
        for key in ("model_sha256","reference_sha256","method","physics_backend","actor_dim","critic_dim"):
            if old[key]!=meta[key]:
                raise ValueError("Checkpoint mismatch: "+key)
        params = pickle.loads(a.checkpoint.read_bytes())
        meta["checkpoint_sha256"] = digest(a.checkpoint)
    elif a.method in ("guided","bc"):
        if a.dataset is None:
            p.error("BC requires successful multi-orientation demonstration trajectories")
        arrays = dataset_arrays(a.dataset, True, digest(model), digest(bank), env.physics_backend)
        params, losses = fit_matched_bc(net, params, arrays, a.bc_updates, a.seed)
        (a.output/"bc_loss.json").write_text(json.dumps(losses, indent=2)+"\n")
        (a.output/"bc_params.pkl").write_bytes(pickle.dumps(jax.device_get(params)))
        meta["dataset_sha256"] = digest(a.dataset)
    sources = a.output/"sources"
    sources.mkdir()
    meta["sources"] = {}
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file and Path(file).resolve().parent==ROOT/"scripts" and file.endswith(".py"):
            file = Path(file).resolve()
            shutil.copyfile(file, sources/file.name)
            meta["sources"][file.name] = digest(file)
    (a.output/"run_config.json").write_text(json.dumps(meta, indent=2)+"\n")
    evaluate = evaluator(env, net, reference=reference_mode)
    seeds = np.arange(a.eval_start,a.eval_start+a.eval_count)
    rows = []
    started = time.monotonic()

    def callback(step, _policy, current):
        batches = [jax.device_get(evaluate(current, jp.asarray(seeds[i:i+a.eval_batch], jp.int32)))
                   for i in range(0,len(seeds),a.eval_batch)]
        initial, trace = jax.tree_util.tree_map(lambda *xs: np.concatenate(xs), *batches)
        assert np.isfinite(initial).all() and all(np.isfinite(v).all() for v in trace.values())
        row, episodes = summarize(trace, seeds)
        assert row["released_rope_force_n"] == 0
        row.update(step=int(step),elapsed_sec=time.monotonic()-started)
        rows.append(row)
        folder = a.output/f"step_{int(step):09d}"
        folder.mkdir(exist_ok=False)
        (folder/"episodes.json").write_text(json.dumps(episodes, indent=2)+"\n")
        np.savez_compressed(folder/"initial_states.npz", qpos=initial, seeds=seeds)
        # Prefer one success and one failure in addition to fixed early previews.
        indices = list(range(min(2,len(seeds))))
        for success in (True,False):
            indices += [i for i,e in enumerate(episodes) if e["success30"]==success and i not in indices][:1]
        np.savez_compressed(folder/"preview_trajectories.npz", seeds=seeds[indices],
                            **{k:v[indices] for k,v in trace.items()})
        if a.method not in ("reference","collect"):
            canonical = jax.tree_util.tree_map(np.asarray, jax.device_get(current))
            (folder/"params.pkl").write_bytes(pickle.dumps(canonical))
            (a.output/"params.pkl").write_bytes(pickle.dumps(canonical))
        with (a.output/"metrics.csv").open("w") as f:
            w = csv.DictWriter(f,fieldnames=list(row)); w.writeheader(); w.writerows(rows)
        print(json.dumps(row),flush=True)
        if a.method=="collect":
            selected = np.array([e["success30"] for e in episodes])
            if selected.sum()<4:
                raise ValueError("Fewer than 4 successful demos; fix reference coverage before BC")
            path = a.output/"trajectories.npz"
            np.savez_compressed(path, **{k:trace[k][selected] for k in ("obs","action","active")})
            manifest = dict(model_sha256=digest(model), reference_sha256=digest(bank),dataset_sha256=digest(path),
                actor_dim=123,physics_backend=env.physics_backend, seeds=seeds[selected].tolist(),
                attempted_seeds=seeds.tolist(), coverage_rate=float(selected.mean()),
                criterion="S09 released 30s palmar thumb/opposing-finger grasp, zero rope force")
            (a.output/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")

    def progress(step, metrics):
        with (a.output/"ppo_metrics.jsonl").open("a") as f:
            f.write(json.dumps(dict(step=int(step),**{k:float(v) for k,v in metrics.items()}))+"\n")

    if a.steps:
        train.train(environment=env,num_timesteps=a.steps,num_envs=32,num_eval_envs=a.eval_batch,
            episode_length=800,batch_size=32,num_minibatches=4,unroll_length=20,num_updates_per_batch=4,
            num_resets_per_eval=0,learning_rate=a.learning_rate,entropy_cost=.0001,discounting=.99,
            normalize_observations=False,reward_scaling=1,network_factory=network_factory,
            wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training,full_reset=True),
            seed=a.seed,num_evals=a.evals,run_evals=False,restore_params=params,restore_value_fn=True,
            policy_params_fn=callback,progress_fn=progress)
    else:
        callback(0,None,params)
    summary = dict(status="completed", rows=rows, final_step=rows[-1]["step"])
    for key in ("success30","hold_sec"):
        best=max(rows,key=lambda r:r[key])
        summary[key]=dict(first=rows[0][key],last=rows[-1][key],max=best[key],best_step=best["step"])
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("DONE",json.dumps(summary),flush=True)


if __name__ == "__main__":
    main()
