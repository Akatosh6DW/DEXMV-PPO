"""Source-matched BC/PPO/guided/residual study with paired tactile inputs."""

import argparse
import csv
import functools
import json
import os
from pathlib import Path
import pickle
from types import SimpleNamespace
import sys
import time

from brax.training.agents.ppo import networks, train
import jax
import jax.numpy as jp
import numpy as np
import optax

from mujoco_playground._src import wrapper
from orca_guidance_recheck import score
from orca_hic53_env import check_hic53_gates
from orca_matched_env import OrcaMatchedGrasp
from orca_refit_env import BASE_WEIGHTS
from orca_residual_controls import load_reference
from prepare_hic_orca_object import digest
from train_orca_comparison import initialize, network_factory
from train_orca_guidance_recheck import curve_summary

ROOT = Path(__file__).resolve().parents[1]


def compare_initial_trace(run, trace, meta):
    prior = json.loads((run / "run_config.json").read_text())
    if prior.get("reference_mode", "video") != meta.get("reference_mode", "video"):
        raise ValueError("Initial replay reference mode mismatch")
    if prior.get("residual_bound", .08) != meta.get("residual_bound", .08):
        raise ValueError("Initial replay residual bound mismatch")
    for key in ("method", "tactile", "seed", "eval_start", "eval_count", "model_sha256",
                "reference_sha256", "actor_dim", "critic_dim", "init_std", "weights", "xla_flags"):
        if prior[key] != meta[key]:
            raise ValueError("Initial replay configuration mismatch: " + key)
    path = run / "step_000000000/trajectories.npz"
    differences = {}
    with np.load(path) as previous:
        if set(previous.files) != set(trace):
            raise ValueError("Initial replay trace fields differ")
        for key, value in trace.items():
            if value.shape != previous[key].shape or not np.isfinite(previous[key]).all():
                raise ValueError("Invalid previous initial trace: " + key)
            differences[key] = float(np.max(np.abs(value.astype(float) - previous[key].astype(float))))
    return dict(exact_match=all(v == 0 for v in differences.values()), differences=differences,
                previous_trace=str(path.resolve()), previous_trace_sha256=digest(path))


def initialize_matched(env, seed, std):
    obs = dict(state=jp.zeros(123), privileged_state=jp.zeros(140))
    proxy = SimpleNamespace(action_size=17, reset=lambda key: SimpleNamespace(obs=obs))
    net, params = initialize(proxy, seed, std)
    head = params[1]["params"]["hidden_2"]
    head["kernel"] = head["kernel"].at[:, :17].set(0.)
    mean = jp.zeros(17) if env.arm == "residual" else jp.arctanh(
        jp.clip(env.ctrl_to_action(jp.asarray(env._config.initial_ctrl)), -.995, .995))
    head["bias"] = head["bias"].at[:17].set(mean)
    return net, params


def make_evaluator(env, net, seeds, teacher_controls=None):
    inference = networks.make_inference_fn(net)
    keys = jax.vmap(jax.random.PRNGKey)(jp.asarray(seeds, jp.int32))
    teacher = None if teacher_controls is None else jp.asarray(teacher_controls)
    def evaluate(params):
        policy = inference(params, deterministic=True)
        def episode(key):
            state = env.reset(key)
            assert state.obs["state"].shape == (123,) and state.obs["privileged_state"].shape == (140,)
            key = jax.random.fold_in(key, 987654)
            def step(carry, _):
                state, alive, key = carry
                key, subkey = jax.random.split(key)
                if teacher is None:
                    action = policy(state.obs, subkey)[0]
                else:
                    action = env.ctrl_to_action(teacher[jp.minimum(state.info["step"], len(teacher) - 1)])
                nxt = env.step(state, action)
                next_alive = alive & (nxt.done == 0)
                trace = dict(obs=state.obs["state"], action=action, active=alive, alive=next_alive,
                             hold=jp.where(next_alive, nxt.metrics["hold_sec"], 0),
                             reward=jp.where(alive, nxt.reward, 0),
                             qpos=nxt.data.qpos, qvel=nxt.data.qvel, ctrl=nxt.data.ctrl,
                             mocap=nxt.data.mocap_pos, forbidden=nxt.metrics["forbidden_contact"],
                             contacts=nxt.metrics["contact_count"], thumb=nxt.metrics["thumb_contact"],
                             released=nxt.metrics["released"], cube_z=nxt.metrics["cube_z"],
                             spawn_z=nxt.info["cube_spawn_pos"][2], cube_speed=nxt.metrics["cube_speed"],
                             finger_normal_force=nxt.info["finger_normal_force"],
                             touch_obs=state.info["touch_obs"], actuator_force_rms=nxt.metrics["actuator_force_rms"])
                for name in ("video_base", "requested_residual", "effective_residual", "desired_ctrl"):
                    trace[name] = nxt.info[name]
                for name in env.weights:
                    trace["component_" + name] = nxt.metrics["component_" + name]
                return (nxt, next_alive, key), trace
            return jax.lax.scan(step, (state, jp.bool_(True), key), None, length=800)[1]
        return jax.vmap(episode)(keys)
    return jax.jit(evaluate)


def dataset_arrays(path, tactile, model_hash, reference_hash, physics_backend="mjx_jax"):
    manifest = json.loads((path.parent / "dataset_manifest.json").read_text())
    if (manifest["dataset_sha256"] != digest(path) or manifest["model_sha256"] != model_hash
            or manifest["reference_sha256"] != reference_hash or manifest["actor_dim"] != 123
            or manifest.get("physics_backend", "mjx_jax") != physics_backend):
        raise ValueError("Demonstration dataset is not bound to this task/reference")
    with np.load(path) as z:
        obs, action, active = z["obs"].copy(), z["action"], z["active"].astype(bool)
    if (obs.ndim != 3 or obs.shape[-1] != 123 or len(obs) < 4
            or active.shape != obs.shape[:2] or action.shape != (*obs.shape[:2], 17)):
        raise ValueError("Invalid matched demonstration shapes")
    if not tactile:
        obs[..., -10:] = 0
    if (not np.isfinite(obs).all() or not np.isfinite(action).all() or np.any(obs[..., 94] != 0)
            or np.any(np.abs(action) > 1.000001)):
        raise ValueError("Invalid data or privileged hold-counter leak")
    split = max(2, len(obs) * 3 // 4)
    early = np.arange(obs.shape[1])[None, :] < 30
    train_mask = np.arange(len(obs))[:, None] < split
    early_mask, late_mask = active & train_mask & early, active & train_mask & ~early
    validation = active & ~train_mask
    if not all(mask.any() for mask in (early_mask, late_mask, validation)):
        raise ValueError("Missing acquisition, holding, or validation samples")
    return tuple(jp.asarray(x) for x in (obs[early_mask], action[early_mask], obs[late_mask],
                                        action[late_mask], obs[validation], action[validation]))


def fit_matched_bc(net, params, arrays, updates, seed):
    early_obs, early_action, late_obs, late_action, val_obs, val_action = arrays
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params[1])
    def loss(policy, obs, actions):
        logits = net.policy_network.apply(params[0], policy, {"state": obs})
        return jp.mean((logits[..., :17] - jp.arctanh(jp.clip(actions, -.995, .995))) ** 2)
    @jax.jit
    def update(policy, opt_state, key):
        a, b = jax.random.split(key)
        i = jax.random.randint(a, (128,), 0, len(early_obs))
        j = jax.random.randint(b, (128,), 0, len(late_obs))
        value, grads = jax.value_and_grad(loss)(policy, jp.concatenate([early_obs[i], late_obs[j]]),
                                               jp.concatenate([early_action[i], late_action[j]]))
        change, opt_state = optimizer.update(grads, opt_state, policy)
        return optax.apply_updates(policy, change), opt_state, value
    validation_loss = jax.jit(lambda policy: loss(policy, val_obs, val_action))
    policy, key, rows = params[1], jax.random.PRNGKey(seed + 5000), []
    for step in range(updates):
        key, subkey = jax.random.split(key)
        policy, opt_state, value = update(policy, opt_state, subkey)
        if step % 100 == 0 or step == updates - 1:
            rows.append(dict(step=step + 1, train_loss=float(value), validation_loss=float(validation_loss(policy))))
    return (params[0], policy, params[2]), rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("collect", "run"), default="run")
    p.add_argument("--method", choices=("bc", "ppo", "guided", "residual"), required=True)
    p.add_argument("--tactile", choices=("off", "on"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--tactile-gate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--bc-checkpoint", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--weights", type=Path)
    p.add_argument("--seed", type=int, default=201)
    p.add_argument("--steps", type=int)
    p.add_argument("--evals", type=int, default=5)
    p.add_argument("--eval-start", type=int, default=9200)
    p.add_argument("--eval-count", type=int, default=16)
    p.add_argument("--bc-updates", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--std", type=float)
    p.add_argument("--residual-bound", type=float, default=.08)
    p.add_argument("--reference-mode", choices=("video", "endpoint", "initial"), default="video")
    p.add_argument("--vectorized-cone-hessian", action="store_true")
    p.add_argument("--native-physics", action="store_true")
    p.add_argument("--compare-initial-to", type=Path,
                   help="Before any PPO update, require exact replay of another run's initial evaluation")
    args = p.parse_args()
    if not np.isfinite(args.residual_bound) or not 0 < args.residual_bound <= .15:
        p.error("Residual bound must be in (0, .15] radians")
    if args.method != "residual" and args.residual_bound != .08:
        p.error("Only residual policies may override the residual bound")
    if args.method != "residual" and args.reference_mode != "video":
        p.error("Only residual policies may override the reference mode")
    if args.native_physics and args.vectorized_cone_hessian:
        p.error("Native physics does not use the MJX Hessian implementation")
    if args.native_physics:
        # Brax uses device_put_replicated, whose mesh conflicts with callback
        # lowering under JAX 0.9's new pmap implementation. Keep legacy pmap.
        jax.config.update("jax_pmap_shmap_merge", False)
    physics_backend = "native_mujoco_callback" if args.native_physics else "mjx_jax"
    args.steps = (0 if args.method == "bc" or args.mode == "collect" else 327680) if args.steps is None else args.steps
    if args.steps < 0 or args.eval_count < 1 or (args.steps and args.evals < 2) or args.bc_updates < 1:
        p.error("Invalid budget")
    if args.method == "bc" and args.steps:
        p.error("BC-only may not perform PPO updates")
    if args.checkpoint and (args.steps or args.bc_checkpoint):
        p.error("checkpoint is evaluation-only; optimizer continuation is not implemented")
    if args.bc_checkpoint and args.method != "guided":
        p.error("Only guided PPO restores a BC initialization")
    if args.mode == "collect" and (args.method != "ppo" or args.tactile != "on"):
        p.error("Collect with method=ppo,tactile=on; teacher bypasses the policy and off data are masked copies")
    if args.mode == "collect" and (args.steps or args.checkpoint or args.bc_checkpoint):
        p.error("Collection does not run PPO or load a trained policy")
    if args.mode == "run" and args.method in ("bc", "guided") and not (args.dataset or args.checkpoint or args.bc_checkpoint):
        p.error("A matched dataset or initialization is required")
    check_hic53_gates(args.model, args.reference, args.gate)
    tactile_gate = json.loads(args.tactile_gate.read_text())
    runtime = None
    if args.vectorized_cone_hessian:
        from orca_solver_runtime import install
        runtime = install()
        runtime["source_sha256"] = digest(ROOT / "scripts/orca_solver_runtime.py")
    if (not tactile_gate["pass_paired_tactile"] or tactile_gate["backend"] != "gpu"
            or tactile_gate["model_sha256"] != digest(args.model)
            or tactile_gate["reference_sha256"] != digest(args.reference)
            or tactile_gate["gate_sha256"] != digest(args.gate)
            or tactile_gate["xla_flags"] != os.environ.get("XLA_FLAGS", "")
            or tactile_gate["jax_version"] != jax.__version__
            or tactile_gate.get("solver_runtime") != runtime
            or tactile_gate.get("physics_backend", "mjx_jax") != physics_backend
            or tactile_gate.get("residual_bound", .08) != args.residual_bound
            or tactile_gate.get("reference_mode", "video") != args.reference_mode
            or (tactile_gate.get("matched_source_sha256") is not None
                and tactile_gate["matched_source_sha256"] != digest(ROOT / "scripts/orca_matched_env.py"))
            or (args.residual_bound != .08 and not tactile_gate.get("bound_control_check", {}).get("passed", False))
            or tactile_gate.get("pmap_shmap_merge", True) != bool(jax.config.jax_pmap_shmap_merge)
            or (args.native_physics and (tactile_gate["native_source_sha256"] != digest(ROOT / "scripts/orca_native_physics.py")
                                         or not tactile_gate.get("pmap_wrapper_check", {}).get("pass_full_reset_wrapper", False)))):
        raise ValueError("Matched training requires the paired tactile gate and its verified runtime configuration")
    args.output.mkdir(parents=True, exist_ok=False)
    assert jax.default_backend() == "gpu"
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "artifacts/orca30/jax_cache"))
    weights = json.loads(args.weights.read_text()) if args.weights else BASE_WEIGHTS
    arm = "residual" if args.method == "residual" else "pure"
    env_type = OrcaMatchedGrasp
    if args.native_physics:
        from orca_native_physics import OrcaNativeMatchedGrasp
        env_type = OrcaNativeMatchedGrasp
    env = env_type(args.model, arm, args.reference, weights, tactile=args.tactile == "on",
                   residual_bound=args.residual_bound, reference_mode=args.reference_mode)
    std = args.std if args.std is not None else (.03 if args.method in ("bc", "guided") else .7)
    net, params = initialize_matched(env, args.seed, std)
    started = time.monotonic()
    meta = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    meta.update(model_sha256=digest(args.model), reference_sha256=digest(args.reference), gate_sha256=digest(args.gate),
                tactile_gate_sha256=digest(args.tactile_gate),
                dataset_sha256=digest(args.dataset) if args.dataset else None,
                weights=weights, init_std=std, actor_dim=123, critic_dim=140,
                initialization="zero residual or constant reset-pose absolute control; shared hidden/value weights",
                tactile_sensor=dict(noise_std_n=.02, filter_alpha=.5, delay_steps=1, force_scale_n=5.),
                source_video_count=1, source_type="published video tracking fits; no RGB estimator rerun",
                xla_flags=os.environ.get("XLA_FLAGS", ""), jax_version=jax.__version__,
                solver_runtime=runtime,
                physics_backend=physics_backend,
                pmap_shmap_merge=bool(jax.config.jax_pmap_shmap_merge),
                bc_elapsed_sec=0., command=sys.argv, backend=jax.default_backend(), source_sha256={})
    sources = args.output / "sources"
    sources.mkdir()
    for name in ("train_orca_matched.py", "orca_matched_env.py", "orca_hic53_env.py", "orca_refit_env.py",
                 "orca_residual_env.py", "orca_benchmark_env.py", "orca_residual_controls.py", "orca_residual_metrics.py",
                 "train_orca_comparison.py", "orca_guidance_recheck.py", "train_orca_guidance_recheck.py",
                 "prepare_hic_orca_object.py", "orca_contact_forces.py", "orca_solver_runtime.py", "orca_native_physics.py"):
        source = ROOT / "scripts" / name
        (sources / name).write_bytes(source.read_bytes())
        meta["source_sha256"][name] = digest(source)
    if args.bc_checkpoint:
        prior = json.loads((args.bc_checkpoint.parent / "run_config.json").read_text())
        if prior.get("physics_backend", "mjx_jax") != physics_backend:
            raise ValueError("BC checkpoint physics backend mismatch")
        for key in ("seed", "tactile", "model_sha256", "reference_sha256", "actor_dim", "critic_dim", "init_std"):
            if prior[key] != meta[key]:
                raise ValueError("BC checkpoint mismatch: " + key)
        if prior["method"] != "bc":
            raise ValueError("Guided initialization must be the paired BC-only checkpoint")
        params = pickle.loads(args.bc_checkpoint.read_bytes())
        meta["bc_checkpoint_sha256"] = digest(args.bc_checkpoint)
        meta["dataset_sha256"] = prior["dataset_sha256"]
        meta["bc_elapsed_sec"] = prior["bc_elapsed_sec"]
        meta["initialization"] = "exact paired BC-only checkpoint; optimizer newly initialized for PPO"
    elif args.checkpoint:
        config = args.checkpoint.parent / "run_config.json"
        if not config.exists():
            config = args.checkpoint.parent.parent / "run_config.json"
        prior = json.loads(config.read_text())
        if prior.get("physics_backend", "mjx_jax") != physics_backend:
            raise ValueError("Use the separate native transfer evaluator for cross-backend checkpoints")
        for key in ("method", "tactile", "model_sha256", "reference_sha256", "actor_dim", "critic_dim"):
            if prior[key] != meta[key]:
                raise ValueError("Evaluation checkpoint mismatch: " + key)
        if prior.get("residual_bound", .08) != args.residual_bound:
            raise ValueError("Evaluation checkpoint residual bound mismatch")
        if prior.get("reference_mode", "video") != args.reference_mode:
            raise ValueError("Evaluation checkpoint reference mode mismatch")
        params = pickle.loads(args.checkpoint.read_bytes())
        meta["checkpoint_sha256"] = digest(args.checkpoint)
        meta["initialization"] = "evaluation-only restored policy checkpoint"
    elif args.mode == "run" and args.method in ("bc", "guided"):
        arrays = dataset_arrays(args.dataset, args.tactile == "on", meta["model_sha256"], meta["reference_sha256"], physics_backend)
        before = time.monotonic()
        params, losses = fit_matched_bc(net, params, arrays, args.bc_updates, args.seed)
        meta["bc_elapsed_sec"] = time.monotonic() - before
        (args.output / "bc_loss.json").write_text(json.dumps(losses, indent=2) + "\n")
        (args.output / "bc_params.pkl").write_bytes(pickle.dumps(jax.device_get(params)))
    (args.output / "run_config.json").write_text(json.dumps(meta, indent=2) + "\n")
    (args.output / "env_config.json").write_text(env._config.to_json(indent=2))
    seeds = list(range(args.eval_start, args.eval_start + args.eval_count))
    teacher = None
    source_frames = None
    if args.mode == "collect":
        with np.load(args.reference) as packet:
            source_frames = [int(packet["source_frame_start"]), len(packet["source_time"]) - 1]
        teacher, _ = load_reference(args.reference, env._lowers, env._uppers, env._config.initial_ctrl,
                                     env.dt, env._config.episode_length, env._config.max_ctrl_delta)
    evaluate = make_evaluator(env, net, seeds, teacher)
    rows = []
    def callback(step, _policy, current):
        canonical = jax.tree_util.tree_map(np.asarray, jax.device_get(current))
        trace = jax.device_get(evaluate(canonical))
        assert all(np.isfinite(v).all() for v in trace.values())
        row, holds, successes = score(trace, env)
        final_valid = trace["alive"][:, -1] & (trace["hold"][:, -1] > 0)
        row.update(success15=float(np.mean((holds >= 15) & final_valid)),
                   success15_any=float(np.mean(holds >= 15)),
                   active_force_max_n=float(np.max(trace["finger_normal_force"][trace["active"]])),
                   step=int(step), elapsed_sec=time.monotonic() - started)
        rows.append(row)
        directory = args.output / f"step_{int(step):09d}"
        directory.mkdir()
        np.savez_compressed(directory / "trajectories.npz", **trace)
        (directory / "params.pkl").write_bytes(pickle.dumps(canonical))
        (directory / "episodes.json").write_text(json.dumps(dict(seeds=seeds, hold_sec=holds.tolist(), success30=successes.tolist())))
        if step == 0 and args.compare_initial_to:
            comparison = compare_initial_trace(args.compare_initial_to, trace, meta)
            (args.output / "initial_runtime_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
            if not comparison["exact_match"]:
                raise AssertionError("Initial physics replay changed; PPO updates are not authorized")
        with (args.output / "metrics.csv").open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)
        if args.mode == "collect":
            np.savez_compressed(args.output / "trajectories.npz", **trace)
            dataset_meta = dict(model_sha256=meta["model_sha256"], reference_sha256=meta["reference_sha256"],
                                dataset_sha256=digest(args.output / "trajectories.npz"), actor_dim=123,
                                seeds=seeds, source_video_count=1, active_samples=int(trace["active"].sum()),
                                source_frames=source_frames, teacher="retargeted video controls, not a PPO expert",
                                physics_backend=physics_backend,
                                note="Perturbed simulator rollouts of ONE video; post-termination frames excluded by BC loader")
            (args.output / "dataset_manifest.json").write_text(json.dumps(dataset_meta, indent=2) + "\n")
    def progress(step, metrics):
        values = {k: float(v) for k, v in metrics.items()}
        assert np.isfinite(list(values.values())).all()
        with (args.output / "ppo_metrics.jsonl").open("a") as file:
            file.write(json.dumps(dict(step=int(step), **values)) + "\n")
    if args.mode == "collect" or not args.steps:
        callback(0, None, params)
    else:
        train.train(environment=env, num_timesteps=args.steps, num_envs=128, num_eval_envs=args.eval_count,
                    episode_length=800, batch_size=128, num_minibatches=4, unroll_length=20,
                    num_updates_per_batch=4, num_resets_per_eval=0, learning_rate=args.learning_rate,
                    entropy_cost=.001, discounting=.99, normalize_observations=False, reward_scaling=1,
                    network_factory=network_factory,
                    wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
                    seed=args.seed, num_evals=args.evals, run_evals=False, restore_params=params, restore_value_fn=True,
                    policy_params_fn=callback, progress_fn=progress)
    summary = curve_summary(rows)
    summary.update(method=args.method, tactile=args.tactile, bc_elapsed_sec=meta["bc_elapsed_sec"])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
