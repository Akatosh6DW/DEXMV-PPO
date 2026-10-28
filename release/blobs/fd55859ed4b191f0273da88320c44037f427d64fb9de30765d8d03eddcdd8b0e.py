"""Collect matched elastic demonstrations and train/evaluate S12 Brax policies."""

import argparse
import copy
import csv
from dataclasses import asdict
import functools
import hashlib
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

from orca_s12_env import OrcaS12Grasp
from s12_soft import SoftConfig, BASE, REFERENCE
from train_orca_comparison import initialize, network_factory
from train_orca_matched import fit_matched_bc


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_digest(env):
    return hashlib.sha256(json.dumps(dict(config=asdict(env.soft_cfg),
        reward=env._config.reward_version, residual_bound=env.residual_bound), sort_keys=True).encode()).hexdigest()


def initialize_soft(env, seed, std):
    net, params = initialize(env, seed, std)
    head = params[1]["params"]["hidden_2"]
    head["kernel"] = head["kernel"].at[:, :17].set(0.)
    mean = jp.zeros(17) if env.arm == "residual" else jp.arctanh(
        jp.clip(env.ctrl_to_action(jp.zeros(17)), -.995, .995))
    head["bias"] = head["bias"].at[:17].set(mean)
    return net, params


def evaluate_serial(env, net, params, seeds, reference=False):
    policy = jax.jit(networks.make_inference_fn(net)(params, deterministic=True))
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    episodes, traces = [], []
    for seed in seeds:
        key = jax.random.PRNGKey(int(seed))
        state = reset(key)
        rows = []
        for i in range(env._config.episode_length):
            action = env.ctrl_to_action(env.reference_base(state.info)) if reference else policy(state.obs, key)[0]
            nxt = step(state, action)
            data, metrics, obs, action_np, info = jax.device_get((nxt.data, nxt.metrics, state.obs["state"], action, nxt.info))
            row = dict(obs=obs, action=action_np, time=(i+1)*env.dt, qpos=data.qpos, qvel=data.qvel,
                ctrl=data.ctrl, mocap=data.mocap_pos, mocap_quat=data.mocap_quat, vertices=data.vertices,
                loads=data.audit[:5], forbidden_force_n=data.audit[8], max_principal_strain=float(data.audit[9]),
                min_volume_ratio=float(data.audit[10]), penetration_m=float(data.audit[5]),
                support_force_n=float(data.audit[7]), peak_force_n=float(data.audit[6]),
                hold_sec=float(metrics["hold_sec"]), success15=float(metrics["success15"]),
                reward=float(nxt.reward), alive=not bool(nxt.done))
            for field in ("video_base", "requested_residual", "desired_ctrl", "offset"):
                row[field] = info[field]
            rows.append(row)
            state = nxt
            if bool(nxt.done):
                break
        trace = {k:np.array([r[k] for r in rows]) for k in rows[0]}
        episodes.append(dict(seed=int(seed), success15=bool(rows[-1]["success15"]),
            best_hold_sec=float(trace["hold_sec"].max()), duration_sec=float(trace["time"][-1]),
            return_sum=float(trace["reward"].sum()), max_principal_strain=float(trace["max_principal_strain"].max()),
            max_penetration_m=float(trace["penetration_m"].max()), peak_force_n=float(trace["peak_force_n"].max()),
            min_volume_ratio=float(trace["min_volume_ratio"].min()),
            final_support_force_n=float(trace["support_force_n"][-1])))
        traces.append(trace)
        print("EPISODE", json.dumps(episodes[-1]), flush=True)
    return episodes, traces


def evaluate(env, net, params, seeds, reference=False):
    if env.native.pool is None:
        return evaluate_serial(env, net, params, seeds, reference)
    policy = jax.jit(jax.vmap(networks.make_inference_fn(net)(params, deterministic=True)))
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    reference_action = jax.jit(jax.vmap(lambda info:env.ctrl_to_action(env.reference_base(info))))
    episodes, traces = [], []
    batch_size = len(env.native.pool.processes)
    for begin in range(0, len(seeds), batch_size):
        group = seeds[begin:begin+batch_size]
        keys = jax.vmap(jax.random.PRNGKey)(jp.asarray(group))
        state = reset(keys)
        initial = state
        active = np.ones(len(group), bool)
        rows = [[] for _ in group]
        for i in range(env._config.episode_length):
            action = reference_action(state.info) if reference else policy(state.obs, keys)[0]
            nxt = step(state, action)
            data, metrics, obs, action_np, reward, done, info = jax.device_get(
                (nxt.data, nxt.metrics, state.obs["state"], action, nxt.reward, nxt.done, nxt.info))
            for j in np.flatnonzero(active):
                rows[j].append(dict(obs=obs[j], action=action_np[j], time=(i+1)*env.dt,
                    qpos=data.qpos[j], qvel=data.qvel[j], ctrl=data.ctrl[j], mocap=data.mocap_pos[j],
                    mocap_quat=data.mocap_quat[j], vertices=data.vertices[j], loads=data.audit[j,:5],
                    forbidden_force_n=float(data.audit[j,8]), max_principal_strain=float(data.audit[j,9]),
                    min_volume_ratio=float(data.audit[j,10]), penetration_m=float(data.audit[j,5]),
                    support_force_n=float(data.audit[j,7]), peak_force_n=float(data.audit[j,6]),
                    hold_sec=float(metrics["hold_sec"][j]), success15=float(metrics["success15"][j]),
                    reward=float(reward[j]), alive=not bool(done[j])))
                for field in ("video_base", "requested_residual", "desired_ctrl", "offset"):
                    rows[j][-1][field] = info[field][j]
            active &= ~done.astype(bool)
            if not active.any():
                break
            # Inactive slots replay only the first safe step, never continue a
            # dropped/degenerated body and never append it to episode statistics.
            mask = jp.asarray(active)
            state = jax.tree_util.tree_map(lambda new, old:jp.where(
                mask.reshape((len(group),)+(1,)*(new.ndim-1)), new, old), nxt, initial)
        for seed, records in zip(group, rows):
            trace = {k:np.array([r[k] for r in records]) for k in records[0]}
            episode = dict(seed=int(seed), success15=bool(records[-1]["success15"]),
                best_hold_sec=float(trace["hold_sec"].max()), duration_sec=float(trace["time"][-1]),
                return_sum=float(trace["reward"].sum()), max_principal_strain=float(trace["max_principal_strain"].max()),
                max_penetration_m=float(trace["penetration_m"].max()), peak_force_n=float(trace["peak_force_n"].max()),
                min_volume_ratio=float(trace["min_volume_ratio"].min()),
                final_support_force_n=float(trace["support_force_n"][-1]))
            episodes.append(episode)
            traces.append(trace)
            print("EPISODE", json.dumps(episode), flush=True)
    return episodes, traces


def enrich_episode(episode, trace, cfg):
    """Separate physical retention from the stricter contact/material gate."""
    center = trace["vertices"].mean(1)
    physical = ((trace["time"]>=cfg.release_sec+.5)&trace["alive"].astype(bool)
        &(center[:,2]>=.535)&(trace["support_force_n"]<1e-5)
        &(trace["loads"][:,0]>.02)&np.any(trace["loads"][:,1:]>.02, axis=1))
    hold = best = 0
    for valid in physical:
        hold = hold+1 if valid else 0
        best = max(best, hold)
    episode.update(physical_success15=bool(hold*cfg.control_dt>=cfg.hold_sec),
        physical_best_hold_sec=best*cfg.control_dt,
        forbidden_force_max_n=float(trace["forbidden_force_n"].max()),
        final_center_z=float(center[-1,2]),
        post_release_strain_max=float(trace["max_principal_strain"][trace["time"]>=cfg.release_sec].max())
            if (trace["time"]>=cfg.release_sec).any() else None,
        requested_residual_rms=float(np.sqrt(np.mean(trace["requested_residual"]**2))))
    return episode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=("collect", "reference", "bc", "guided", "ppo", "residual"), required=True)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--evals", type=int, default=3)
    p.add_argument("--eval-count", type=int, default=4)
    p.add_argument("--eval-start", type=int, default=251000)
    p.add_argument("--seed", type=int, default=25201)
    p.add_argument("--std", type=float, default=.05)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--bc-updates", type=int, default=10000)
    p.add_argument("--bc-phase-only", action="store_true")
    p.add_argument("--jitter", type=float, default=.002)
    p.add_argument("--young", type=float, default=10000.)
    p.add_argument("--timestep", type=float, default=.0002)
    p.add_argument("--hand-offset-z", type=float, default=0.)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--physics-workers", type=int, default=0)
    a = p.parse_args()
    if a.steps and a.method not in ("ppo", "guided", "residual"):
        p.error("Method does not perform PPO updates")
    if a.method in ("guided", "bc") and not (a.checkpoint or a.dataset):
        p.error("Matched soft-body demonstrations required")
    a.output.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_pmap_shmap_merge", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    cfg = SoftConfig(young=a.young, timestep=a.timestep, hand_offset_z=a.hand_offset_z)
    env = OrcaS12Grasp(cfg, "residual" if a.method == "residual" else "pure", jitter=a.jitter)
    if a.physics_workers:
        env.native.enable_workers(a.physics_workers)
    net, params = initialize_soft(env, a.seed, a.std)
    (a.output/"model.xml").write_text(env.native.sim.xml)
    obs = env.reset(jax.random.PRNGKey(0)).obs
    meta = {k:str(v.resolve()) if isinstance(v, Path) else v for k,v in vars(a).items()}
    meta.update(soft_config=asdict(cfg), model_sha256=digest(a.output/"model.xml"),
        source_model_sha256=digest(BASE), reference_sha256=digest(REFERENCE),
        physics_backend=env.physics_backend, actor_dim=int(obs["state"].shape[0]),
        critic_dim=int(obs["privileged_state"].shape[0]), backend=jax.default_backend(),
        reward_version=env._config.reward_version)
    meta["task_sha256"] = task_digest(env)
    if a.checkpoint:
        prior = json.loads((a.checkpoint.parent/"run_config.json").read_text())
        for k in ("model_sha256", "reference_sha256", "actor_dim", "critic_dim", "reward_version", "task_sha256"):
            if prior[k] != meta[k]:
                raise ValueError("Checkpoint mismatch: "+k)
        if prior["method"] != a.method and not (prior["method"] == "bc" and a.method == "guided"):
            raise ValueError("Checkpoint action interpretation differs")
        params = pickle.loads(a.checkpoint.read_bytes())
        meta["checkpoint_sha256"] = digest(a.checkpoint)
    elif a.method in ("bc", "guided"):
        manifest = json.loads((a.dataset.parent/"dataset_manifest.json").read_text())
        if digest(a.dataset) != manifest["dataset_sha256"]:
            raise ValueError("Dataset hash mismatch")
        for k in ("model_sha256", "reference_sha256", "actor_dim", "physics_backend", "task_sha256"):
            if manifest[k] != meta[k]:
                raise ValueError("Demo mismatch: "+k)
        with np.load(a.dataset) as z:
            observations, actions = z["obs"], z["action"]
        expected = (len(observations), env._config.episode_length)
        if (observations.shape!=(*expected, meta["actor_dim"])
                or actions.shape!=(*expected, env.action_size)
                or not np.isfinite(observations).all() or not np.isfinite(actions).all()
                or np.max(np.abs(actions))>1.000001):
            raise ValueError("Invalid matched demonstration shapes/values")
        if len(observations)<4:
            raise ValueError("Need at least four successful independent demonstrations")
        split = max(2, len(observations)*3//4)
        idx = np.arange(observations.shape[1])[None, :]
        is_train = np.arange(len(observations))[:, None]<split
        masks = (is_train&(idx<160), is_train&(idx>=160), ~is_train&np.ones_like(idx, bool))
        arrays = tuple(jp.asarray(v[mask]) for mask in masks for v in (observations, actions))
        if a.bc_phase_only:
            # This teacher is a time-indexed reference, not a recovery controller.
            # Match S10: initialize BC with phase only, then let PPO train all inputs.
            mask = np.zeros(meta["actor_dim"], bool)
            mask[87:90] = True
            params = copy.deepcopy(params)
            first = params[1]["params"]["hidden_0"]
            first["kernel"] = first["kernel"]*jp.asarray(mask)[:, None]
            arrays = tuple(v*jp.asarray(mask) if i%2==0 else v for i,v in enumerate(arrays))
            meta["bc_input_columns"] = np.flatnonzero(mask).tolist()
        params, losses = fit_matched_bc(net, copy.deepcopy(params), arrays, a.bc_updates, a.seed)
        (a.output/"bc_loss.json").write_text(json.dumps(losses, indent=2)+"\n")
        (a.output/"bc_params.pkl").write_bytes(pickle.dumps(jax.device_get(params)))
        meta["dataset_sha256"] = digest(a.dataset)
    sources = a.output/"sources"
    sources.mkdir()
    meta["sources"] = {}
    for module in list(sys.modules.values()):
        f = getattr(module, "__file__", None)
        if f and Path(f).resolve().parent == ROOT/"scripts" and f.endswith(".py"):
            src = Path(f).resolve()
            shutil.copyfile(src, sources/src.name)
            meta["sources"][src.name] = digest(src)
    (a.output/"run_config.json").write_text(json.dumps(meta, indent=2)+"\n")
    started = time.monotonic()
    rows = []
    seeds = np.arange(a.eval_start, a.eval_start+a.eval_count)

    def callback(step, _policy, current):
        episodes, traces = evaluate(env, net, current, seeds, a.method in ("collect", "reference"))
        episodes = [enrich_episode(e, t, cfg) for e,t in zip(episodes, traces)]
        folder = a.output/f"step_{int(step):09d}"
        folder.mkdir(exist_ok=False)
        (folder/"episodes.json").write_text(json.dumps(episodes, indent=2)+"\n")
        for i, trace in enumerate(traces):
            np.savez_compressed(folder/f"trajectory_{int(seeds[i])}.npz", **trace)
        if a.method not in ("collect", "reference"):
            payload = pickle.dumps(jax.device_get(current))
            (folder/"params.pkl").write_bytes(payload)
            (a.output/"params.pkl").write_bytes(payload)
        row = dict(step=int(step), elapsed_sec=time.monotonic()-started,
            success15=float(np.mean([e["success15"] for e in episodes])),
            hold_sec=float(np.mean([e["best_hold_sec"] for e in episodes])),
            return_mean=float(np.mean([e["return_sum"] for e in episodes])),
            strain_max=max(e["max_principal_strain"] for e in episodes),
            penetration_max=max(e["max_penetration_m"] for e in episodes),
            force_max=max(e["peak_force_n"] for e in episodes))
        rows.append(row)
        with (a.output/"metrics.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader(); writer.writerows(rows)
        print("EVAL", json.dumps(row), flush=True)
        if a.method == "collect":
            selected = [(e, t) for e,t in zip(episodes, traces) if e["success15"]]
            if len(selected)<4:
                raise ValueError("Insufficient successful soft-body demonstrations")
            dataset = a.output/"trajectories.npz"
            np.savez_compressed(dataset, obs=np.stack([t["obs"] for e,t in selected]),
                                action=np.stack([t["action"] for e,t in selected]))
            manifest = {k:meta[k] for k in ("model_sha256", "reference_sha256", "actor_dim", "physics_backend", "task_sha256")}
            manifest.update(dataset_sha256=digest(dataset), attempted_seeds=seeds.tolist(),
                            seeds=[e["seed"] for e,t in selected])
            (a.output/"dataset_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")

    def progress(step, metrics):
        with (a.output/"ppo_metrics.jsonl").open("a") as f:
            f.write(json.dumps(dict(step=int(step), **{k:float(v) for k,v in metrics.items()}))+"\n")

    if a.steps:
        train.train(environment=env, num_timesteps=a.steps, num_envs=8, num_eval_envs=1,
            episode_length=env._config.episode_length, batch_size=8, num_minibatches=4, unroll_length=20,
            num_updates_per_batch=4, num_resets_per_eval=0, learning_rate=a.learning_rate,
            entropy_cost=.0001, discounting=.99, normalize_observations=False, reward_scaling=1,
            network_factory=network_factory, wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
            seed=a.seed, num_evals=a.evals, run_evals=False, restore_params=params, restore_value_fn=True,
            policy_params_fn=callback, progress_fn=progress)
    else:
        callback(0, None, params)
    summary = dict(status="completed", rows=rows)
    for metric in ("success15", "hold_sec"):
        best = max(rows, key=lambda row:row[metric])
        summary[metric] = dict(first=rows[0][metric], last=rows[-1][metric], max=best[metric], best_step=best["step"])
    (a.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("DONE", json.dumps(summary), flush=True)
    if env.native.pool is not None:
        env.native.pool.close()


if __name__ == "__main__":
    main()
