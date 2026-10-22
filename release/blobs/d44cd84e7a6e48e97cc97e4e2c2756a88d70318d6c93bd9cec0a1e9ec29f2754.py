#!/usr/bin/env python3
"""Brax PPO with paired initialization and independent no-autoreset evaluation."""

import argparse
import csv
import functools
import hashlib
import json
from pathlib import Path
import pickle
import time

from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks, train
import jax
import jax.numpy as jp
import numpy as np
import optax
import mujoco

from mujoco_playground._src import wrapper
from orca_benchmark_env import OrcaBenchmark


def network_factory(observation_size, action_size, **kwargs):
    return networks.make_ppo_networks(
        observation_size, action_size, **kwargs, policy_hidden_layer_sizes=(128, 128),
        value_hidden_layer_sizes=(128, 128), policy_obs_key="state",
        value_obs_key="privileged_state", init_noise_std=.2,
        mean_kernel_init_fn=jax.nn.initializers.lecun_uniform)


def initialize(env, seed, init_std=None):
    shape = {k: v.shape for k, v in env.reset(jax.random.PRNGKey(0)).obs.items()}
    net = network_factory(observation_size=shape, action_size=env.action_size)
    global_key, _ = jax.random.split(jax.random.PRNGKey(seed))
    kp, kv = jax.random.split(global_key)
    stats = running_statistics.init_state({k: specs.Array(v, jp.float32) for k, v in shape.items()})
    policy = net.policy_network.init(kp)
    if init_std is not None:
        if init_std <= .001:
            raise ValueError("init_std must exceed the tanh-normal minimum .001")
        # Brax's tanh_normal MLP ignores init_noise_std. Initialize its scale
        # outputs explicitly and equally; BC gradients only affect the mean.
        head = policy["params"]["hidden_2"]
        if head["bias"].shape != (2 * env.action_size,):
            raise ValueError("Unexpected policy head shape")
        raw_scale = np.log(np.expm1(init_std - .001))
        head["kernel"] = head["kernel"].at[:, env.action_size:].set(0)
        head["bias"] = head["bias"].at[env.action_size:].set(raw_scale)
    return net, (stats, policy, net.value_network.init(kv))


def fit_bc(net, params, dataset, steps, seed):
    # Balanced acquisition/holding sampling avoids a constant hold policy
    # dominating the loss merely because the episode contains more hold frames.
    with np.load(dataset) as z:
        obs = np.asarray(z["obs"], np.float32)
        acts = np.asarray(z["action"], np.float32)
    early_obs = jp.asarray(obs[:, :60].reshape(-1, obs.shape[-1]))
    early_act = jp.asarray(acts[:, :60].reshape(-1, acts.shape[-1]))
    late_obs = jp.asarray(obs[:, 60:].reshape(-1, obs.shape[-1]))
    late_act = jp.asarray(acts[:, 60:].reshape(-1, acts.shape[-1]))
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params[1])

    def loss(policy, observations, targets):
        logits = net.policy_network.apply(params[0], policy, {"state": observations})
        means = logits[..., :17]
        return jp.mean((means - jp.arctanh(jp.clip(targets, -.995, .995))) ** 2)

    @jax.jit
    def update(policy, opt_state, key):
        k1, k2 = jax.random.split(key)
        i = jax.random.randint(k1, (128,), 0, len(early_obs))
        j = jax.random.randint(k2, (128,), 0, len(late_obs))
        observations = jp.concatenate([early_obs[i], late_obs[j]])
        targets = jp.concatenate([early_act[i], late_act[j]])
        value, grads = jax.value_and_grad(loss)(policy, observations, targets)
        updates, opt_state = optimizer.update(grads, opt_state, policy)
        return optax.apply_updates(policy, updates), opt_state, value

    policy = params[1]
    losses = []
    key = jax.random.PRNGKey(seed + 5000)
    for i in range(steps):
        key, subkey = jax.random.split(key)
        policy, opt_state, value = update(policy, opt_state, subkey)
        if i % 100 == 0 or i == steps - 1:
            losses.append(dict(step=i + 1, loss=float(value)))
    return (params[0], policy, params[2]), losses


def make_demo_regularized_loss(original, observations, actions, weight):
    observations, actions = jp.asarray(observations), jp.asarray(actions)
    def loss(params, normalizer_params, data, rng, *, ppo_network, **kwargs):
        value, metrics = original(params, normalizer_params, data, rng,
                                  ppo_network=ppo_network, **kwargs)
        indices = jax.random.randint(jax.random.fold_in(rng, 81), (256,), 0, len(observations))
        logits = ppo_network.policy_network.apply(
            normalizer_params, params.policy, {"state": observations[indices]})
        target = jp.arctanh(jp.clip(actions[indices], -.995, .995))
        imitation = jp.mean((logits[..., :17] - target) ** 2)
        total = value + weight * imitation
        return total, {**metrics, "ppo_loss": value, "demo_loss": imitation,
                       "demo_weighted_loss": weight * imitation, "total_loss": total}
    return loss


def make_evaluator(env, net, seeds):
    keys = jax.vmap(jax.random.PRNGKey)(jp.array(seeds, dtype=jp.int32))
    inference = networks.make_inference_fn(net)

    def evaluate(params):
        policy = inference(params, deterministic=True)

        def episode(key):
            state = env.reset(key)
            def step(carry, _):
                state, alive = carry
                action, _ = policy(state.obs, key)
                nxt = env.step(state, action)
                alive &= nxt.done == 0
                return (nxt, alive), dict(
                    hold=jp.where(alive, nxt.metrics["hold_sec"], 0),
                    alive=alive, qpos=nxt.data.qpos, mocap=nxt.data.mocap_pos,
                    qvel=nxt.data.qvel,
                    ctrl=nxt.data.ctrl, action=action,
                    forbidden=nxt.metrics["forbidden_contact"],
                    contacts=nxt.metrics["contact_count"],
                    thumb=nxt.metrics["thumb_contact"],
                    released=nxt.metrics["released"], cube_z=nxt.metrics["cube_z"],
                    spawn_z=nxt.info["cube_spawn_pos"][2],
                    reward=jp.where(alive, nxt.reward, 0))
            return jax.lax.scan(step, (state, jp.bool_(True)), None, length=800)[1]
        return jax.vmap(episode)(keys)
    return jax.jit(evaluate)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("scratch", "bc"), required=True)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--steps", type=int, default=327680)
    p.add_argument("--evals", type=int, default=5)
    p.add_argument("--bc-steps", type=int, default=2000)
    p.add_argument("--dataset", type=Path, default=Path("artifacts/orca30/mjx_o05/trajectories.npz"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--eval-start", type=int, default=100)
    p.add_argument("--eval-count", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--confirmatory", action="store_true")
    p.add_argument("--init-std", type=float)
    p.add_argument("--demo-weight", type=float, default=0.0)
    args = p.parse_args()
    if args.demo_weight < 0 or (args.arm == "scratch" and args.demo_weight != 0):
        p.error("A nonnegative demonstration loss is available only in the BC arm")
    args.output.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_compilation_cache_dir", str(Path(__file__).resolve().parents[1] / "artifacts/orca30/jax_cache"))
    started = time.monotonic()
    env = OrcaBenchmark()
    net, params = initialize(env, args.seed, args.init_std)
    bc_time = 0
    if args.arm == "bc":
        t = time.monotonic()
        params, losses = fit_bc(net, params, args.dataset, args.bc_steps, args.seed)
        bc_time = time.monotonic() - t
        (args.output / "bc_loss.json").write_text(json.dumps(losses, indent=2))
    meta = vars(args).copy()
    meta = {k: str(v) if isinstance(v, Path) else v for k, v in meta.items()}
    meta.update(bc_sec=bc_time, dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest()
                if args.arm == "bc" else None, development=not args.confirmatory,
                source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ("train_orca_comparison.py", "orca_benchmark_env.py")})
    if args.arm == "bc" and (args.dataset.parent / "report.json").exists():
        meta["demonstration_provenance"] = json.loads((args.dataset.parent / "report.json").read_text())
    mujoco.mj_saveLastXML(str(args.output / "model.xml"), env.mj_model)
    meta["model_sha256"] = hashlib.sha256((args.output / "model.xml").read_bytes()).hexdigest()
    (args.output / "run_config.json").write_text(json.dumps(meta, indent=2))
    (args.output / "env_config.json").write_text(env._config.to_json(indent=2))
    seeds = list(range(args.eval_start, args.eval_start + args.eval_count))
    evaluate = make_evaluator(env, net, seeds)
    rows = []

    def callback(step, make_policy, current):
        del make_policy
        result = jax.device_get(evaluate(current))
        hold = result["hold"].max(axis=1)
        row = dict(step=int(step), success30=float(np.mean(hold >= 30)),
                   mean_hold=float(np.mean(hold)), min_hold=float(np.min(hold)),
                   max_hold=float(np.max(hold)),
                   mean_return=float(np.mean(result["reward"].sum(axis=1))),
                   elapsed_sec=time.monotonic() - started, bc_sec=bc_time)
        rows.append(row)
        with (args.output / "metrics.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        checkpoint = args.output / f"step_{int(step):09d}"
        checkpoint.mkdir(exist_ok=True)
        with (checkpoint / "params.pkl").open("wb") as f:
            pickle.dump(jax.device_get(current), f)
        (checkpoint / "episodes.json").write_text(json.dumps(dict(seeds=seeds, hold_sec=hold.tolist()), indent=2))
        np.savez_compressed(checkpoint / "trajectories.npz", **result)
        print(json.dumps(row), flush=True)

    def progress(step, metrics):
        with (args.output / "ppo_metrics.jsonl").open("a") as f:
            f.write(json.dumps(dict(step=int(step), **{k: float(v) for k, v in metrics.items()})) + "\n")

    original_loss = ppo_losses.compute_ppo_loss
    if args.demo_weight:
        with np.load(args.dataset) as z:
            obs, act = z["obs"], z["action"]
        # Keep equal acquisition/hold representation in the regularizer.
        obs = np.concatenate([obs[:, :60], obs[:, 60::13]], axis=1).reshape(-1, obs.shape[-1])
        act = np.concatenate([act[:, :60], act[:, 60::13]], axis=1).reshape(-1, act.shape[-1])
        ppo_losses.compute_ppo_loss = make_demo_regularized_loss(original_loss, obs, act, args.demo_weight)
    try:
      train.train(environment=env, num_timesteps=args.steps,
                num_envs=128, num_eval_envs=args.eval_count, episode_length=800,
                batch_size=128, num_minibatches=4, unroll_length=20,
                num_updates_per_batch=4, num_resets_per_eval=0,
                learning_rate=args.learning_rate, entropy_cost=1e-3,
                discounting=.99, normalize_observations=False, reward_scaling=1,
                network_factory=network_factory,
                wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
                seed=args.seed, num_evals=args.evals, run_evals=False,
                restore_params=params, restore_value_fn=True,
                policy_params_fn=callback, progress_fn=progress)
    finally:
        ppo_losses.compute_ppo_loss = original_loss
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
