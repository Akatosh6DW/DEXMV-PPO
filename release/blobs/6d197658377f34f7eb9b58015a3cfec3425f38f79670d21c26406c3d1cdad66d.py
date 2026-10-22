#!/usr/bin/env python3
"""Matched pure/residual PPO with audited, independent rollout evaluation."""

import argparse
import csv
import functools
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import time
from types import SimpleNamespace

from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks, train
import jax
import jax.numpy as jp
import mujoco
import numpy as np

from mujoco_playground._src import wrapper
from orca_residual_env import OrcaResidualGrasp
from orca_residual_metrics import summarize
from train_orca_comparison import fit_bc, initialize, make_demo_regularized_loss, network_factory

ROOT = Path(__file__).resolve().parents[1]


def initialize_zero_mean(seed, std=.7):
    obs = dict(state=jp.zeros(113), privileged_state=jp.zeros(130))
    proxy = SimpleNamespace(action_size=17, reset=lambda key: SimpleNamespace(obs=obs))
    net, params = initialize(proxy, seed, std)
    head = params[1]["params"]["hidden_2"]
    head["kernel"] = head["kernel"].at[:, :17].set(0)
    head["bias"] = head["bias"].at[:17].set(0)
    return net, params


def make_evaluator(env, net, seeds, zero=False):
    keys = jax.vmap(jax.random.PRNGKey)(jp.asarray(seeds, dtype=jp.int32))
    inference = networks.make_inference_fn(net)

    def evaluate(params):
        policy = inference(params, deterministic=True)

        def episode(key):
            state = env.reset(key)
            if state.obs["state"].shape != (113,) or state.obs["privileged_state"].shape != (130,):
                raise ValueError("Observation contract changed")

            def step(carry, _):
                state, alive = carry
                action = jp.zeros(17) if zero else policy(state.obs, key)[0]
                nxt = env.step(state, action)
                alive &= nxt.done == 0
                trace = dict(obs=state.obs["state"],
                             hold=jp.where(alive, nxt.metrics["hold_sec"], 0),
                             alive=alive, qpos=nxt.data.qpos, qvel=nxt.data.qvel,
                             mocap=nxt.data.mocap_pos, ctrl=nxt.data.ctrl, action=action,
                             forbidden=nxt.metrics["forbidden_contact"],
                             contacts=nxt.metrics["contact_count"], thumb=nxt.metrics["thumb_contact"],
                             released=nxt.metrics["released"], cube_z=nxt.metrics["cube_z"],
                             spawn_z=nxt.info["cube_spawn_pos"][2],
                             reward=jp.where(alive, nxt.reward, 0))
                for name in ("video_base", "requested_residual", "effective_residual", "desired_ctrl"):
                    trace[name] = nxt.info[name]
                return (nxt, alive), trace
            return jax.lax.scan(step, (state, jp.bool_(True)), None, length=800)[1]
        return jax.vmap(episode)(keys)
    return jax.jit(evaluate)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=("pure", "bc", "residual"))
    p.add_argument("--shape", required=True, choices=("cube", "sphere", "cuboid"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=ROOT/"artifacts/orca30/dexmv_slow_acquisition_o21/reference.npz")
    p.add_argument("--seed", type=int, default=90)
    p.add_argument("--steps", type=int, default=163840)
    p.add_argument("--evals", type=int, default=3)
    p.add_argument("--eval-start", type=int, default=100)
    p.add_argument("--eval-count", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--init-std", type=float, default=.7)
    p.add_argument("--bound", type=float, default=.08)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--bc-steps", type=int, default=2000)
    p.add_argument("--demo-weight", type=float, default=1000.)
    p.add_argument("--zero", action="store_true", help="Evaluate zero actions only; no training")
    p.add_argument("--checkpoint", type=Path, help="Evaluate a trusted local checkpoint only")
    p.add_argument("--confirmatory", action="store_true")
    args = p.parse_args()
    if args.zero and args.checkpoint:
        p.error("Choose zero actions or a checkpoint")
    if args.arm == "bc" and not args.checkpoint and args.dataset is None:
        p.error("BC requires the shared video-derived dataset")
    if args.zero and args.arm != "residual":
        p.error("The prior-only baseline uses the residual environment")
    args.output.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", str(ROOT/"artifacts/orca30/jax_cache"))
    started = time.monotonic()
    env = OrcaResidualGrasp(ROOT/f"artifacts/orca_residual/models/{args.shape}.xml",
                           "pure" if args.arm == "bc" else args.arm, args.reference, args.bound)
    net, params = initialize_zero_mean(args.seed, args.init_std)
    if args.checkpoint:
        with args.checkpoint.open("rb") as f:
            params = pickle.load(f)
    bc_sec = 0.
    if args.arm == "bc" and not args.checkpoint:
        before = time.monotonic()
        params, losses = fit_bc(net, params, args.dataset, args.bc_steps, args.seed)
        bc_sec = time.monotonic()-before
        (args.output/"bc_loss.json").write_text(json.dumps(losses, indent=2))
    meta = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    meta.update(command=sys.argv, precision=jax.config.jax_default_matmul_precision,
                development=not args.confirmatory, backend=jax.default_backend(),
                jax_version=jax.__version__, mujoco_version=mujoco.__version__,
                initialization="zero mean head, same random hidden/value weights per seed",
                primary_success="continuous clean hold >=30s and valid at final 40s frame",
                reference_type="time-indexed retargeted video trajectory; not visual feedback",
                bc_steps=args.bc_steps if args.arm == "bc" else 0,
                bc_sec=bc_sec, demonstration_loss_weight=args.demo_weight if args.arm == "bc" else 0)
    if args.arm == "bc" and args.dataset:
        meta["dataset_sha256"] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
        meta["dataset_provenance"] = json.loads((args.dataset.parent/"report.json").read_text())
    sources = args.output/"sources"
    sources.mkdir()
    names = ("train_orca_residual.py", "orca_residual_env.py", "orca_residual_controls.py",
             "orca_residual_metrics.py", "orca_benchmark_env.py", "train_orca_comparison.py")
    meta["source_sha256"] = {}
    for name in names:
        source = Path(__file__).with_name(name)
        shutil.copy2(source, sources/name)
        meta["source_sha256"][name] = hashlib.sha256(source.read_bytes()).hexdigest()
    mujoco.mj_saveLastXML(str(args.output/"model.xml"), env.mj_model)
    meta["model_sha256"] = hashlib.sha256((args.output/"model.xml").read_bytes()).hexdigest()
    meta["reference_sha256"] = None
    if args.arm == "residual":
        meta["reference_sha256"] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
        shutil.copy2(args.reference, args.output/"reference.npz")
        report = args.reference.parent/"report.json"
        if report.exists():
            meta["reference_provenance"] = json.loads(report.read_text())
        np.savez_compressed(args.output/"processed_reference.npz",
                            ctrl=np.asarray(env.reference_ctrl), raw_ctrl=np.asarray(env.raw_reference_ctrl))
    (args.output/"run_config.json").write_text(json.dumps(meta, indent=2))
    (args.output/"env_config.json").write_text(env._config.to_json(indent=2))
    seeds = list(range(args.eval_start, args.eval_start+args.eval_count))
    evaluate = make_evaluator(env, net, seeds, args.zero)
    rows = []

    def callback(step, make_policy, current):
        del make_policy
        result = jax.device_get(evaluate(current))
        row, holds, success = summarize(result, args.arm, args.bound, env._lowers, env._uppers)
        np.testing.assert_array_less(np.abs(result["ctrl"][:, 0]-np.asarray(env._config.initial_ctrl)),
                                     env._config.max_ctrl_delta+1e-5)
        if args.zero and args.arm == "residual":
            np.testing.assert_allclose(result["ctrl"], result["video_base"], atol=1e-5)
        row = dict(step=int(step), **row, elapsed_sec=time.monotonic()-started)
        rows.append(row)
        with (args.output/"metrics.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        checkpoint = args.output/f"step_{int(step):09d}"
        checkpoint.mkdir(exist_ok=False)
        with (checkpoint/"params.pkl").open("wb") as f:
            pickle.dump(jax.device_get(current), f)
        np.savez_compressed(checkpoint/"trajectories.npz", **result)
        (checkpoint/"episodes.json").write_text(json.dumps(dict(seeds=seeds, hold_sec=holds.tolist(),
                                                               success30=success.tolist()), indent=2))
        print(json.dumps(row), flush=True)

    def progress(step, metrics):
        with (args.output/"ppo_metrics.jsonl").open("a") as f:
            f.write(json.dumps(dict(step=int(step), **{k: float(v) for k, v in metrics.items()}))+"\n")

    if args.zero or args.checkpoint or args.steps == 0:
        callback(0, None, params)
    else:
        original_loss = ppo_losses.compute_ppo_loss
        if args.arm == "bc":
            with np.load(args.dataset, allow_pickle=False) as z:
                obs, act = z["obs"], z["action"]
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
    scores = np.array([r["success30"] for r in rows])
    summary = dict(first=float(scores[0]), last=float(scores[-1]), max=float(scores.max()),
                   best_step=rows[int(np.argmax(scores))]["step"], evaluations=len(rows))
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2))
    print("DONE", args.output, summary, flush=True)


if __name__ == "__main__":
    main()
