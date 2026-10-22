"""Development PPO for the source-matched passive cap, with saved evaluations."""

import argparse
import csv
import functools
import json
import os
from pathlib import Path
import pickle
import time

from brax.training.agents.ppo import networks, train
import jax
import jax.numpy as jp
import mujoco
import numpy as np

from mujoco_playground._src import wrapper
from oakink_cap_env import OakInkCap
from prepare_hic_orca_object import digest
from train_orca_comparison import initialize, network_factory
from train_orca_matched import fit_matched_bc


def initialize_cap(env, seed, std):
    net, params = initialize(env, seed, std)
    head = params[1]["params"]["hidden_2"]
    head["kernel"] = head["kernel"].at[:, :17].set(0)
    mean = jp.zeros(17) if env.method == "residual" else jp.arctanh(jp.clip(env.ctrl_to_action(env.initial), -.995, .995))
    head["bias"] = head["bias"].at[:17].set(mean)
    return net, params


def make_evaluator(env, net, seeds, teacher=False):
    inference = networks.make_inference_fn(net)
    keys = jax.vmap(jax.random.PRNGKey)(jp.asarray(seeds, jp.int32))

    def evaluate(params):
        policy = inference(params, deterministic=True)

        def episode(key):
            state = env.reset(key)

            def step(carry, _):
                state, alive = carry
                if teacher == "initial":
                    action = env.ctrl_to_action(env.initial)
                elif teacher:
                    action = env.ctrl_to_action(env.next_base(state.info["step"]))
                else:
                    action = policy(state.obs, key)[0]
                nxt = env.step(state, action)
                next_alive = alive & (nxt.done == 0)
                trace = dict(obs=state.obs["state"], action=action, active=alive, alive=next_alive,
                             reward=jp.where(alive, nxt.reward, 0), qpos=nxt.data.qpos, qvel=nxt.data.qvel,
                             ctrl=nxt.data.ctrl, finger_force=nxt.data.finger_force, cap_center=nxt.data.cap_center,
                             thread_active=nxt.data.thread_active, cap_fingers=nxt.data.cap_fingers,
                             bottle_contact=nxt.data.bottle_contact, floor_contact=nxt.data.floor_contact,
                             reset_qpos=state.info["reset_qpos"], reset_penetration=state.info["reset_penetration"])
                trace.update({name: nxt.metrics[name] for name in nxt.metrics})
                trace.update({name: nxt.info[name] for name in
                              ("video_base", "requested_residual", "effective_residual", "desired_ctrl")})
                return (nxt, next_alive), trace

            state.info["reset_qpos"] = state.data.qpos
            state.info["reset_penetration"] = state.data.hand_penetration
            return jax.lax.scan(step, (state, jp.bool_(True)), None, length=400)[1]
        return jax.vmap(episode)(keys)
    return jax.jit(evaluate)


def dataset_arrays(path, env):
    manifest = json.loads((path.parent / "dataset_manifest.json").read_text())
    if (digest(path) != manifest["dataset_sha256"] or digest(env.xml_path) != manifest["model_sha256"]
            or env.reference_sha256 != manifest["reference_sha256"]):
        raise ValueError("Cap dataset does not match model/reference")
    with np.load(path) as z:
        obs, action, active = z["obs"].copy(), z["action"], z["active"].astype(bool)
    if obs.shape != (32, 400, 124) or action.shape != (32, 400, 17) or active.shape != (32, 400):
        raise ValueError("Expected 32 matched cap teacher episodes")
    if not np.isfinite(obs).all() or not np.isfinite(action).all() or np.abs(action).max() > 1.000001:
        raise ValueError("Invalid cap teacher data")
    np.testing.assert_array_equal(obs[..., -27:-10], 0)
    if not env.tactile:
        obs[..., -10:] = 0
    train_mask = np.arange(32)[:, None] < 24
    early = np.arange(400)[None, :] < 100
    masks = [active & train_mask & early, active & train_mask & ~early, active & ~train_mask]
    if not all(mask.any() for mask in masks):
        raise ValueError("Empty cap BC train/validation stratum")
    return tuple(jp.asarray(value[mask]) for mask in masks for value in (obs, action))


def summarize(trace):
    alive, active = trace["alive"].astype(bool), trace["active"].astype(bool)
    success = (trace["success"][:, -1] > 0) & alive[:, -1]
    released = np.any((trace["released"] > 0) & alive, axis=1)
    return dict(success=float(success.mean()), release_rate=float(released.mean()),
                mean_max_degrees=float(np.max(np.where(active, trace["degrees"], -np.inf), axis=1).mean()),
                mean_final_degrees=float(np.array([a[m][-1] for a, m in zip(trace["degrees"], active)]).mean()),
                mean_best_hold_s=float(np.max(np.where(alive, trace["hold_sec"], 0), axis=1).mean()),
                peak_force_n=float(trace["force"][active].max()),
                max_penetration_mm=float(trace["hand_penetration"][active].max() * 1000),
                forbidden_fraction=float(trace["forbidden"][active].mean()),
                mean_return=float(trace["reward"].sum(axis=1).mean()),
                terminal_failure_rate=float((~alive[:, -1]).mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=("pure", "residual", "bc", "guided"), required=True)
    p.add_argument("--collect", action="store_true")
    p.add_argument("--dataset", type=Path)
    p.add_argument("--bc-checkpoint", type=Path)
    p.add_argument("--bc-updates", type=int, default=2000)
    p.add_argument("--protocol", type=Path, help="Frozen cap comparison protocol; omitted for development")
    p.add_argument("--tactile", choices=("on", "off"), default="on")
    p.add_argument("--steps", type=int)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--evals", type=int, default=3)
    p.add_argument("--eval-count", type=int, default=4)
    p.add_argument("--eval-start", type=int, default=18400)
    p.add_argument("--seed", type=int, default=1801)
    p.add_argument("--std", type=float, default=.7)
    p.add_argument("--learning-rate", type=float, default=.0003)
    args = p.parse_args()
    args.steps = (0 if args.collect or args.method == "bc" else 16384) if args.steps is None else args.steps
    if args.steps < 0 or (args.steps and args.evals < 2) or args.eval_count < 1 or args.std <= .001 or args.bc_updates < 1:
        p.error("Invalid development training budget")
    if args.collect and (args.method != "pure" or args.steps or args.eval_count != 32 or args.tactile != "on"):
        p.error("Collect with pure, tactile on, zero training steps, and 32 resets")
    if args.method == "bc" and (args.steps or not args.dataset):
        p.error("BC requires dataset and zero PPO steps")
    if args.method == "guided" and not args.bc_checkpoint:
        p.error("Guided PPO requires a paired BC checkpoint")
    if args.bc_checkpoint and args.method != "guided":
        p.error("Only guided PPO may load a BC checkpoint")
    if args.protocol:
        from oakink_cap_protocol import check_protocol, check_recipe
        protocol = check_protocol(args.protocol)
        check_recipe(args, protocol)
    jax.config.update("jax_pmap_shmap_merge", False)
    jax.config.update("jax_compilation_cache_dir", str(Path("artifacts/orca30/jax_cache").resolve()))
    gate = json.loads(args.gate.read_text())
    if not (gate["passed"] and gate["backend"] == jax.default_backend() == "gpu"
            and gate["model_sha256"] == digest(args.model / "model.xml")
            and gate["reference_sha256"] == digest(args.reference)):
        raise ValueError("Training requires the current model/reference GPU gate")
    for name, expected in gate["sources"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError("Gate source changed: " + name)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name in (*gate["sources"], "train_oakink_cap.py", "train_orca_comparison.py", "train_orca_matched.py", "orca_native_physics.py"):
        path = Path(__file__).with_name(name)
        (args.output / name).write_bytes(path.read_bytes())
        sources[name] = digest(path)
    env = OakInkCap(args.model, args.reference, "residual" if args.method == "residual" else "pure", args.tactile == "on")
    env.reference_sha256 = digest(args.reference)
    net, params = initialize_cap(env, args.seed, args.std)
    bc_seconds = 0.
    if args.method == "bc":
        start = time.monotonic()
        params, losses = fit_matched_bc(net, params, dataset_arrays(args.dataset, env), args.bc_updates, args.seed)
        bc_seconds = time.monotonic() - start
        (args.output / "bc_loss.json").write_text(json.dumps(losses, indent=2) + "\n")
        with (args.output / "bc_params.pkl").open("wb") as f:
            pickle.dump(jax.device_get(params), f)
    elif args.method == "guided":
        paired = json.loads((args.bc_checkpoint.parent / "run_config.json").read_text())
        for key, value in dict(method="bc", seed=args.seed, std=args.std, tactile=args.tactile,
                               model_sha256=gate["model_sha256"], reference_sha256=gate["reference_sha256"]).items():
            if paired[key] != value:
                raise ValueError("BC initialization is not paired: " + key)
        with args.bc_checkpoint.open("rb") as f:
            params = pickle.load(f)
    seeds = list(range(args.eval_start, args.eval_start + args.eval_count))
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(development=args.protocol is None, protocol_sha256=digest(args.protocol) if args.protocol else None,
                  model_sha256=gate["model_sha256"], reference_sha256=gate["reference_sha256"],
                  gate_sha256=digest(args.gate), sources=sources, evaluation_seeds=seeds,
                  actor_dim=124, critic_dim=148, xla_flags=os.environ.get("XLA_FLAGS", ""),
                  jax_version=jax.__version__, mujoco_version=mujoco.__version__,
                  bc_seconds=bc_seconds, dataset_sha256=digest(args.dataset) if args.dataset else None,
                  bc_checkpoint_sha256=digest(args.bc_checkpoint) if args.bc_checkpoint else None,
                  mechanism="Passive analytic helix with source-derived release angle; no cap actuator",
                  success="Released, cap clearance >=18mm, paired fingers, no forbidden/support contacts, hold >=1s at final frame")
    mujoco.mj_saveModel(env.mj_model, str(args.output / "model.mjb"))
    config["binary_model_sha256"] = digest(args.output / "model.mjb")
    (args.output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    with (args.output / "initial_params.pkl").open("wb") as f:
        pickle.dump(jax.device_get(params), f)
    evaluate = make_evaluator(env, net, seeds, teacher=args.collect)
    rows, started = [], time.monotonic()

    def callback(step, make_policy, current):
        del make_policy
        trace = jax.device_get(evaluate(current))
        if not all(np.isfinite(v).all() for v in trace.values()):
            raise FloatingPointError("Nonfinite evaluation")
        row = dict(step=int(step), **summarize(trace), elapsed_sec=time.monotonic() - started)
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
        print(json.dumps(row), flush=True)

    def progress(step, metrics):
        with (args.output / "ppo_metrics.jsonl").open("a") as f:
            f.write(json.dumps(dict(step=int(step), **{k: float(v) for k, v in metrics.items()})) + "\n")

    if args.steps:
      train.train(environment=env, num_timesteps=args.steps, num_envs=args.num_envs,
                num_eval_envs=args.eval_count, episode_length=400, batch_size=args.batch_size,
                num_minibatches=4, unroll_length=16, num_updates_per_batch=4, num_resets_per_eval=0,
                learning_rate=args.learning_rate, entropy_cost=.001, discounting=.99,
                normalize_observations=False, reward_scaling=1, network_factory=network_factory,
                wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
                seed=args.seed, num_evals=args.evals, run_evals=False,
                restore_params=params, restore_value_fn=True, policy_params_fn=callback, progress_fn=progress)
    else:
        callback(0, None, params)
    if args.collect:
        source = args.output / "step_000000000/trajectories.npz"
        dataset = args.output / "dataset.npz"
        with np.load(source) as packet:
            np.savez_compressed(dataset, **{key: packet[key] for key in ("obs", "action", "active")})
        (args.output / "dataset_manifest.json").write_text(json.dumps(dict(
            dataset_sha256=digest(dataset), source_trace_sha256=digest(source), model_sha256=gate["model_sha256"],
            reference_sha256=gate["reference_sha256"], seeds=seeds, teacher="Video IK target replay, not a successful expert",
            successful_episodes=rows[0]["success"] * 32, actor_dim=124, train_episodes=24, validation_episodes=8,
            source_video_count=1, no_cap_state_override=True), indent=2) + "\n")
    best = max(rows, key=lambda row: row["success"])
    (args.output / "complete.json").write_text(json.dumps(dict(
        complete=True, development=args.protocol is None, first=rows[0], last=rows[-1], best=best,
        final_checkpoint=f"step_{rows[-1]['step']:09d}",
        final_params_sha256=digest(args.output / f"step_{rows[-1]['step']:09d}/params.pkl")), indent=2) + "\n")


if __name__ == "__main__":
    main()
