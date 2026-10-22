#!/usr/bin/env python3
"""Development PPO runs on a fixed-bottle passive-cap task."""

import argparse
import csv
import functools
import hashlib
import json
from pathlib import Path
import pickle
import time

from brax.training.agents.ppo import networks, train
import jax
import jax.numpy as jp
import mujoco
import numpy as np

from mujoco_playground._src import wrapper
from orca_cap_env import OrcaCap
from orca_cap_metrics import summarize
from train_orca_comparison import fit_bc, initialize, network_factory


def make_evaluator(env, net, seeds):
    inference = networks.make_inference_fn(net)
    keys = jax.vmap(jax.random.PRNGKey)(jp.array(seeds, jp.int32))

    def evaluate(params):
        policy = inference(params, deterministic=True)

        def episode(key):
            state = env.reset(key)

            def step(carry, _):
                state, alive = carry
                action, _ = policy(state.obs, key)
                nxt = env.step(state, action)
                alive &= nxt.done == 0
                return (nxt, alive), dict(qpos=nxt.data.qpos, qvel=nxt.data.qvel,
                                          ctrl=nxt.data.ctrl, action=action, alive=alive,
                                          reward=jp.where(alive, nxt.reward, 0),
                                          **nxt.metrics)
            return jax.lax.scan(step, (state, jp.bool_(True)), None,
                                length=env._config.episode_length)[1]
        return jax.vmap(episode)(keys)
    return jax.jit(evaluate)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=327680)
    p.add_argument("--evals", type=int, default=5)
    p.add_argument("--eval-count", type=int, default=4)
    p.add_argument("--eval-start", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--init-std", type=float, default=.7)
    p.add_argument("--restore", type=Path)
    p.add_argument("--bc-dataset", type=Path)
    p.add_argument("--bc-steps", type=int, default=2000)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.restore and args.bc_dataset:
        p.error("Choose checkpoint continuation or BC initialization, not both")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    env = OrcaCap(args.model)
    net, params = initialize(env, args.seed, args.init_std)
    if args.restore:
        with args.restore.open("rb") as f:
            params = pickle.load(f)
    bc_sec = 0.
    if args.bc_dataset:
        bc_started = time.monotonic()
        params, losses = fit_bc(net, params, args.bc_dataset, args.bc_steps, args.seed)
        bc_sec = time.monotonic() - bc_started
        (args.output / "bc_loss.json").write_text(json.dumps(losses, indent=2))
    mujoco.mj_saveLastXML(str(args.output / "model.xml"), env.mj_model)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(development=True, reward_version="cap_v1", runtime_video_action=False,
                  bc_sec=bc_sec, demo_weight=0.,
                  bc_dataset_sha256=hashlib.sha256(args.bc_dataset.read_bytes()).hexdigest() if args.bc_dataset else None,
                  source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ("train_orca_cap.py", "orca_cap_env.py", "orca_cap_scene.py", "orca_cap_metrics.py")})
    (args.output / "run_config.json").write_text(json.dumps(config, indent=2))
    (args.output / "env_config.json").write_text(env._config.to_json(indent=2))
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for name in config["source_sha256"]:
        (snapshot / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    if args.smoke:
        state = env.reset(jax.random.PRNGKey(args.seed))
        step = jax.jit(env.step)
        for _ in range(10):
            state = step(state, env.ctrl_to_action(jp.zeros(17).at[0].set(-.25)))
        result = dict(obs_shapes={k: list(v.shape) for k, v in state.obs.items()},
                      metrics={k: float(v) for k, v in state.metrics.items()},
                      finite=bool(jp.all(jp.isfinite(state.data.qpos))), elapsed=time.monotonic()-started)
        assert result["finite"] and abs(result["metrics"]["angle"]) < 1e-3
        (args.output / "smoke.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return
    seeds = list(range(args.eval_start, args.eval_start + args.eval_count))
    evaluate = make_evaluator(env, net, seeds)
    rows = []

    def callback(step, make_policy, current):
        del make_policy
        trace = jax.device_get(evaluate(current))
        row, holds = summarize(trace, env)
        row = dict(step=int(step), **row, elapsed_sec=time.monotonic()-started)
        rows.append(row)
        with (args.output / "metrics.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        checkpoint = args.output / f"step_{int(step):09d}"
        checkpoint.mkdir(exist_ok=True)
        with (checkpoint / "params.pkl").open("wb") as f:
            pickle.dump(jax.device_get(current), f)
        np.savez_compressed(checkpoint / "trajectories.npz", **trace)
        (checkpoint / "episodes.json").write_text(json.dumps(dict(seeds=seeds, hold_sec=holds.tolist()), indent=2))
        print(json.dumps(row), flush=True)

    if args.eval_only:
        callback(0, None, params)
        return

    def progress(step, metrics):
        with (args.output / "ppo_metrics.jsonl").open("a") as f:
            f.write(json.dumps(dict(step=int(step), **{k: float(v) for k, v in metrics.items()})) + "\n")

    train.train(environment=env, num_timesteps=args.steps, num_envs=128,
                num_eval_envs=args.eval_count, episode_length=env._config.episode_length,
                batch_size=128, num_minibatches=4, unroll_length=20, num_updates_per_batch=4,
                num_resets_per_eval=0, learning_rate=args.learning_rate, entropy_cost=1e-3,
                discounting=.99, normalize_observations=False, reward_scaling=1,
                network_factory=network_factory,
                wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
                seed=args.seed, num_evals=args.evals, run_evals=False, restore_params=params,
                restore_value_fn=True, policy_params_fn=callback, progress_fn=progress)
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
