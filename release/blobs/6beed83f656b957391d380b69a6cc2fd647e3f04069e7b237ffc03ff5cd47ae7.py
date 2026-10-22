"""S05 training/collection with strict distal-pad metrics and frozen run sources."""

import argparse
import csv
import functools
import json
import os
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

from orca_s05_env import OrcaS05Grasp
from orca_residual_controls import load_reference
from prepare_hic_orca_object import digest
from train_orca_acquisition import initial_gate
from train_orca_comparison import network_factory
from train_orca_matched import initialize_matched, dataset_arrays, fit_matched_bc


def evaluator(env, net, seeds, teacher=None):
    inference = networks.make_inference_fn(net)
    def evaluate(params):
        policy = inference(params, deterministic=True)
        def episode(key):
            state = env.reset(key)
            initial = state.data.qpos
            def step(carry, _):
                state, alive, key = carry
                key, subkey = jax.random.split(key)
                action = (policy(state.obs, subkey)[0] if teacher is None else
                          env.ctrl_to_action(teacher[jp.minimum(state.info["step"], len(teacher) - 1)]))
                nxt = jax.lax.cond(alive, lambda s: env.step(s, action), lambda s: s, state)
                next_alive = alive & (nxt.done == 0)
                m = nxt.metrics
                trace = dict(obs=state.obs["state"], action=action, active=alive, alive=next_alive,
                             hold=jp.where(next_alive, m["pad_hold_sec"], 0), broad_hold=m["hold_sec"],
                             qpos=nxt.data.qpos, qvel=nxt.data.qvel, ctrl=nxt.data.ctrl,
                             mocap=nxt.data.mocap_pos, pad_audit=nxt.data.pad_audit,
                             pad_pair=m["pad_pair"], forbidden=m["forbidden_contact"],
                             contacts=m["contact_count"], thumb=m["thumb_contact"], released=m["released"],
                             cube_z=m["cube_z"], spawn_z=nxt.info["cube_spawn_pos"][2],
                             substep_penetration=m["substep_penetration"],
                             substep_forbidden=m["substep_forbidden"], substep_force_peak=m["substep_force_peak"],
                             requested_residual=nxt.info["requested_residual"],
                             effective_residual=nxt.info["effective_residual"],
                             video_base=nxt.info["video_base"], reward=jp.where(alive, nxt.reward, 0))
                return (nxt, next_alive, key), trace
            trace = jax.lax.scan(step, (state, jp.bool_(True), jax.random.fold_in(key, 987654)), None, length=800)[1]
            return initial, trace
        return jax.vmap(episode)(jax.vmap(jax.random.PRNGKey)(jp.array(seeds, jp.int32)))
    return jax.jit(evaluate)


def summarize(trace, seeds):
    episodes = []
    for i, seed in enumerate(seeds):
        active = trace["active"][i].astype(bool)
        released = active & (trace["released"][i] > 0)
        depth = float(np.max(trace["substep_penetration"][i][released], initial=0))
        forbidden = bool(np.any(trace["substep_forbidden"][i][released] > 0))
        hold = float(np.max(trace["hold"][i]))
        final = bool(trace["alive"][i, -1] and trace["hold"][i, -1] > 0)
        episodes.append(dict(seed=seed, pad_hold_sec=hold, broad_hold_sec=float(trace["broad_hold"][i].max()),
                             success30=hold >= 30 and final and not forbidden and depth <= .003,
                             success15=hold >= 15 and final and not forbidden and depth <= .003,
                             final_valid=final, penetration_m=depth, forbidden=forbidden,
                             pad_pair_fraction=float(np.mean(trace["pad_pair"][i][released])) if released.any() else 0.,
                             peak_force_n=float(np.max(trace["substep_force_peak"][i][active], initial=0)),
                             residual_rms_rad=float(np.sqrt(np.mean(trace["requested_residual"][i][active] ** 2)))))
    row = {k: float(np.mean([e[k] for e in episodes])) for k in
           ("success30", "success15", "pad_hold_sec", "broad_hold_sec", "pad_pair_fraction", "residual_rms_rad")}
    row.update(max_penetration_m=max(e["penetration_m"] for e in episodes),
               forbidden_rate=float(np.mean([e["forbidden"] for e in episodes])),
               peak_force_n=max(e["peak_force_n"] for e in episodes))
    return row, episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("residual", "ppo", "guided", "bc", "reference", "collect"), default="residual")
    parser.add_argument("--steps", type=int, default=20480)
    parser.add_argument("--evals", type=int, default=3)
    parser.add_argument("--eval-count", type=int, default=4)
    parser.add_argument("--eval-start", type=int, default=51000)
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--std", type=float, default=.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--pad-shaping", type=float, default=2.)
    parser.add_argument("--height-weight", type=float, default=0.)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--bc-updates", type=int, default=4000)
    parser.add_argument("--save-all-trajectories", action="store_true")
    args = parser.parse_args()
    if args.method in ("reference", "collect", "bc") and args.steps:
        parser.error("Reference, collection and BC-only jobs require --steps 0")
    args.output.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_pmap_shmap_merge", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "artifacts/orca30/jax_cache"))
    assert jax.default_backend() == "gpu", "Use the authorized host GPU execution environment"
    started = time.monotonic()
    model, reference = args.input / "model.xml", args.input / "reference.npz"
    arm = "residual" if args.method == "residual" else "pure"
    env = OrcaS05Grasp(model, arm, reference, pad_shaping=args.pad_shaping, height_weight=args.height_weight)
    gate = initial_gate(env, 16)
    net, params = initialize_matched(env, args.seed, args.std)
    meta = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    meta.update(model=str(model.resolve()), reference=str(reference.resolve()),
                model_sha256=digest(model), reference_sha256=digest(reference), actor_dim=123, critic_dim=140,
                initial_gate=gate, physics_backend=env.physics_backend, backend=jax.default_backend())
    sources = args.output / "sources"
    sources.mkdir()
    meta["source_sha256"] = {}
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file and Path(file).resolve().parent == ROOT / "scripts" and file.endswith(".py"):
            path = Path(file).resolve()
            shutil.copyfile(path, sources / path.name)
            meta["source_sha256"][path.name] = digest(path)
    if args.checkpoint:
        directory = args.checkpoint.parent
        if not (directory / "run_config.json").exists():
            directory = directory.parent
        old = json.loads((directory / "run_config.json").read_text())
        for field in ("model_sha256", "reference_sha256", "method", "actor_dim", "critic_dim"):
            if old[field] != meta[field]:
                raise ValueError("Checkpoint mismatch: " + field)
        params = pickle.loads(args.checkpoint.read_bytes())
        meta["checkpoint_sha256"] = digest(args.checkpoint)
    elif args.method in ("guided", "bc"):
        if args.dataset is None:
            parser.error("BC requires S05 demonstrations")
        arrays = dataset_arrays(args.dataset, True, digest(model), digest(reference), env.physics_backend)
        params, losses = fit_matched_bc(net, params, arrays, args.bc_updates, args.seed)
        (args.output / "bc_loss.json").write_text(json.dumps(losses, indent=2) + "\n")
        (args.output / "bc_params.pkl").write_bytes(pickle.dumps(jax.device_get(params)))
        meta["dataset_sha256"] = digest(args.dataset)
    (args.output / "run_config.json").write_text(json.dumps(meta, indent=2) + "\n")
    (args.output / "env_config.json").write_text(env._config.to_json(indent=2))
    seeds = list(range(args.eval_start, args.eval_start + args.eval_count))
    teacher = None
    if args.method in ("collect", "reference"):
        teacher, _ = load_reference(reference, env._lowers, env._uppers, env._config.initial_ctrl,
                                     env.dt, 800, env._config.max_ctrl_delta)
        teacher = jp.asarray(teacher)
    evaluate = evaluator(env, net, seeds, teacher)
    rows = []

    def callback(step, _policy, current):
        canonical = jax.tree_util.tree_map(np.asarray, jax.device_get(current))
        initial, trace = jax.device_get(evaluate(canonical))
        assert np.isfinite(initial).all() and all(np.isfinite(v).all() for v in trace.values())
        row, episodes = summarize(trace, seeds)
        row.update(step=int(step), elapsed_sec=time.monotonic() - started)
        rows.append(row)
        directory = args.output / f"step_{int(step):09d}"
        directory.mkdir(exist_ok=False)
        (directory / "episodes.json").write_text(json.dumps(episodes, indent=2) + "\n")
        np.savez_compressed(directory / "initial_states.npz", qpos=initial, seeds=seeds)
        saved_count = len(seeds) if args.save_all_trajectories else 4
        np.savez_compressed(directory / "preview_trajectories.npz", **{k: v[:saved_count] for k, v in trace.items()})
        (directory / "params.pkl").write_bytes(pickle.dumps(canonical))
        (args.output / "params.pkl").write_bytes(pickle.dumps(canonical))
        with (args.output / "metrics.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)
        if args.method == "collect":
            selected = np.array([e["success30"] for e in episodes])
            if selected.sum() < 4:
                raise ValueError("Fewer than four strictly valid 30-second demonstrations")
            path = args.output / "trajectories.npz"
            np.savez_compressed(path, **{k: trace[k][selected] for k in ("obs", "action", "active")})
            manifest = dict(model_sha256=digest(model), reference_sha256=digest(reference),
                            dataset_sha256=digest(path), actor_dim=123, physics_backend=env.physics_backend,
                            seeds=np.array(seeds)[selected].tolist(), criterion=getattr(
                                env, "criterion", "S05 strict distal pads, 30s, substep audited"))
            (args.output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def progress(step, metrics):
        with (args.output / "ppo_metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(dict(step=int(step), **{k: float(v) for k, v in metrics.items()})) + "\n")

    if args.steps:
        train.train(environment=env, num_timesteps=args.steps, num_envs=32, num_eval_envs=len(seeds),
                    episode_length=800, batch_size=32, num_minibatches=4, unroll_length=20,
                    num_updates_per_batch=4, num_resets_per_eval=0, learning_rate=args.learning_rate,
                    entropy_cost=.0001, discounting=.99, normalize_observations=False, reward_scaling=1,
                    network_factory=network_factory,
                    wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
                    seed=args.seed, num_evals=args.evals, run_evals=False, restore_params=params,
                    restore_value_fn=True, policy_params_fn=callback, progress_fn=progress)
    else:
        callback(0, None, params)
    summary = dict(status="completed", rows=rows, final_step=rows[-1]["step"])
    for metric in ("success30", "pad_hold_sec"):
        best = max(rows, key=lambda r: r[metric])
        summary[metric] = dict(first=rows[0][metric], last=rows[-1][metric], max=best[metric], best_step=best["step"])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
