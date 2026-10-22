"""Exploratory cube study with quality-filtered demonstrations and compact logs."""

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
sys.path.insert(0, str(ROOT / "Aero-Hand/sim_rl/mujoco_playground"))

from brax.training.agents.ppo import train
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco_playground._src import wrapper

from orca_acquisition_env import OrcaAcquisitionGrasp
from orca_guidance_recheck import score
from orca_residual_controls import load_reference
from prepare_hic_orca_object import digest
from probe_hic53_orca import cube_penetration
from train_orca_matched import initialize_matched, make_evaluator, fit_matched_bc, dataset_arrays
from train_orca_comparison import network_factory

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/orca_collision_refit/O20_hic53_elliptic/cube.xml"
REFERENCE = ROOT / "artifacts/orca_cube_acquisition/E01_flex50_refinement/candidate_reference.npz"


def initial_gate(env, count=16):
    states = jax.device_get(jax.jit(jax.vmap(env.reset))(
        jax.vmap(jax.random.PRNGKey)(jp.arange(8800, 8800 + count))))
    data = mujoco.MjData(env.mj_model)
    penetrations = []
    for q, m in zip(states.data.qpos, states.data.mocap_pos):
        data.qpos[:] = q
        data.mocap_pos[:] = m
        mujoco.mj_forward(env.mj_model, data)
        penetrations.append(cube_penetration(env.mj_model, data))
    assert max(penetrations) <= .003
    assert states.obs["state"].shape == (count, 123)
    assert np.isfinite(states.obs["state"]).all()
    assert np.all(states.obs["state"][:, 94] == 0)
    return dict(initial_penetration_max_m=max(penetrations), seeds=list(range(8800, 8800+count)),
                actor_dim=123, critic_dim=140, passed=True)


def episode_scores(trace, env, seeds):
    row, holds, successes = score(trace, env)
    final = trace["alive"][:, -1] & (trace["hold"][:, -1] > 0)
    active = trace["active"]
    force = np.where(active[..., None], trace["finger_normal_force"], 0)
    valid = trace["hold"] > 0
    acquired = valid.any(axis=1)
    first = np.where(acquired, (valid.argmax(axis=1)+1)*.05, np.nan)
    peaks = force.max(axis=(1, 2))
    forbidden = np.any((trace["forbidden"] > 0) & active & (trace["released"] > 0), axis=1)
    row.update(success15=float(np.mean((holds >= 15) & final)),
               mean_finger_peak_n=float(peaks.mean()), acquisition_rate=float(acquired.mean()),
               mean_acquisition_sec=float(np.nanmean(first)) if acquired.any() else 40.,
               forbidden_rate=float(forbidden.mean()))
    episodes = [dict(seed=int(s), success30=bool(successes[i]), success15=bool(holds[i]>=15 and final[i]),
                     longest_hold_sec=float(holds[i]), final_valid=bool(final[i]),
                     acquisition_sec=float(first[i]) if acquired[i] else None,
                     finger_peak_n=float(peaks[i]), forbidden=bool(forbidden[i])) for i, s in enumerate(seeds)]
    return row, episodes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("run", "collect", "smoke"), default="run")
    p.add_argument("--method", choices=("ppo", "guided", "residual", "bc"), default="ppo")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=REFERENCE)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--seed", type=int, default=801)
    p.add_argument("--steps", type=int, default=163840)
    p.add_argument("--evals", type=int, default=5)
    p.add_argument("--eval-start", type=int, default=12000)
    p.add_argument("--eval-count", type=int, default=8)
    p.add_argument("--bc-updates", type=int, default=4000)
    p.add_argument("--std", type=float, default=.3)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--jitter", type=float, default=.002)
    p.add_argument("--shaping", type=float, default=1.)
    p.add_argument("--geometry-audit", action="store_true",
                   help="Recheck all active evaluation poses against the physical geometry")
    args = p.parse_args()
    if args.checkpoint and args.steps:
        p.error("Checkpoint is evaluation-only; set --steps 0")
    if args.method == "bc" and args.steps:
        p.error("BC-only requires --steps 0")
    if args.mode != "run" and args.method != "ppo":
        p.error("Collect/smoke use pure control")
    args.output.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_pmap_shmap_merge", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "artifacts/orca30/jax_cache"))
    assert jax.default_backend() == "gpu"
    arm = "residual" if args.method == "residual" else "pure"
    env = OrcaAcquisitionGrasp(MODEL, arm, args.reference, jitter=args.jitter, shaping=args.shaping)
    gate = initial_gate(env)
    (args.output / "initial_gate.json").write_text(json.dumps(gate, indent=2)+"\n")
    net, params = initialize_matched(env, args.seed, args.std)
    started = time.monotonic()
    meta = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    meta.update(model_sha256=digest(MODEL), reference_sha256=digest(args.reference),
                dataset_sha256=digest(args.dataset) if args.dataset else None,
                model=str(MODEL), actor_dim=123, critic_dim=140, physics_backend=env.physics_backend,
                initial_gate=gate, source_video_count=1, source_sequence=20,
                note="Exploratory tuned task; video tracking plus explicitly bounded simulator refinement",
                source_sha256={}, bc_elapsed_sec=0.)
    sources = args.output / "sources"
    sources.mkdir()
    for name in ("train_orca_acquisition.py", "orca_acquisition_env.py", "orca_native_physics.py",
                 "train_orca_matched.py", "orca_matched_env.py", "orca_hic53_env.py", "orca_refit_env.py",
                 "orca_residual_env.py", "orca_benchmark_env.py", "orca_residual_controls.py",
                 "orca_residual_metrics.py", "train_orca_comparison.py", "orca_contact_forces.py"):
        shutil.copyfile(ROOT / "scripts" / name, sources / name)
        meta["source_sha256"][name] = digest(sources / name)
    if args.checkpoint:
        config_path = args.checkpoint.parent / "run_config.json"
        if not config_path.exists():
            config_path = args.checkpoint.parent.parent / "run_config.json"
        prior = json.loads(config_path.read_text())
        for key in ("method", "model_sha256", "reference_sha256", "actor_dim", "critic_dim"):
            if prior[key] != meta[key]:
                raise ValueError("Checkpoint mismatch: " + key)
        params = pickle.loads(args.checkpoint.read_bytes())
        meta["checkpoint_sha256"] = digest(args.checkpoint)
    elif args.method in ("bc", "guided"):
        if args.dataset is None:
            p.error("BC/guided require a quality-filtered dataset")
        arrays = dataset_arrays(args.dataset, True, meta["model_sha256"], meta["reference_sha256"], env.physics_backend)
        before = time.monotonic()
        params, losses = fit_matched_bc(net, params, arrays, args.bc_updates, args.seed)
        meta["bc_elapsed_sec"] = time.monotonic()-before
        (args.output / "bc_loss.json").write_text(json.dumps(losses, indent=2)+"\n")
        (args.output / "bc_params.pkl").write_bytes(pickle.dumps(jax.device_get(params)))
    (args.output / "run_config.json").write_text(json.dumps(meta, indent=2)+"\n")
    (args.output / "env_config.json").write_text(env._config.to_json(indent=2))
    seeds = list(range(args.eval_start, args.eval_start + args.eval_count))
    teacher = None
    if args.mode in ("collect", "smoke"):
        teacher, _ = load_reference(args.reference, env._lowers, env._uppers, env._config.initial_ctrl,
                                     env.dt, 800, env._config.max_ctrl_delta)
    evaluate = make_evaluator(env, net, seeds, teacher)
    rows = []

    def callback(step, _policy, current):
        canonical = jax.tree_util.tree_map(np.asarray, jax.device_get(current))
        trace = jax.device_get(evaluate(canonical))
        assert all(np.isfinite(v).all() for v in trace.values())
        row, episodes = episode_scores(trace, env, seeds)
        if args.geometry_audit:
            data = mujoco.MjData(env.mj_model)
            for i, episode in enumerate(episodes):
                maximum = 0.
                for q, m in zip(trace["qpos"][i][trace["active"][i]], trace["mocap"][i][trace["active"][i]]):
                    data.qpos[:] = q
                    data.mocap_pos[:] = m
                    mujoco.mj_forward(env.mj_model, data)
                    maximum = max(maximum, cube_penetration(env.mj_model, data))
                episode["max_penetration_m"] = maximum
            row["max_penetration_m"] = max(e["max_penetration_m"] for e in episodes)
        row.update(step=int(step), elapsed_sec=time.monotonic()-started)
        rows.append(row)
        directory = args.output / f"step_{int(step):09d}"
        directory.mkdir(exist_ok=False)
        (directory / "episodes.json").write_text(json.dumps(episodes, indent=2)+"\n")
        # Preserve all independent episode statistics, only four visual trajectories.
        np.savez_compressed(directory / "preview_trajectories.npz", **{k: v[:4] for k, v in trace.items()})
        (args.output / "params.pkl").write_bytes(pickle.dumps(canonical))
        (directory / "params.pkl").write_bytes(pickle.dumps(canonical))
        with (args.output / "metrics.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)
        if args.mode == "collect":
            chosen = np.array([r["success30"] and not r["forbidden"] for r in episodes])
            if chosen.sum() < 4:
                raise ValueError("Fewer than four successful physical demonstrations")
            # Geometry check every saved active pose before admitting BC samples.
            data = mujoco.MjData(env.mj_model)
            penetration = []
            for i in np.flatnonzero(chosen):
                maximum = 0.
                for q, m in zip(trace["qpos"][i], trace["mocap"][i]):
                    data.qpos[:] = q
                    data.mocap_pos[:] = m
                    mujoco.mj_forward(env.mj_model, data)
                    maximum = max(maximum, cube_penetration(env.mj_model, data))
                penetration.append(dict(seed=seeds[i], max_penetration_m=maximum))
                if maximum > .003:
                    chosen[i] = False
            if chosen.sum() < 4:
                raise ValueError("Insufficient geometrically valid successful demonstrations")
            path = args.output / "trajectories.npz"
            np.savez_compressed(path, **{k: trace[k][chosen] for k in ("obs", "action", "active")})
            manifest = dict(model_sha256=meta["model_sha256"], reference_sha256=meta["reference_sha256"],
                            dataset_sha256=digest(path), actor_dim=123, physics_backend=env.physics_backend,
                            source_video_count=1, source_sequence=20, source_frames=[30,95],
                            attempts=len(seeds), accepted=int(chosen.sum()), seeds=np.array(seeds)[chosen].tolist(),
                            penetration=penetration, teacher="video retargeting plus <=.12 rad simulator-selected correction",
                            split="first 75% rollouts training, remainder validation; not independent videos")
            (args.output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")

    def progress(step, metrics):
        values = {k: float(v) for k, v in metrics.items()}
        assert np.isfinite(list(values.values())).all()
        with (args.output / "ppo_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(step=int(step), **values))+"\n")

    if args.mode != "run" or not args.steps:
        callback(0, None, params)
    else:
        train.train(environment=env, num_timesteps=args.steps, num_envs=128, num_eval_envs=args.eval_count,
                    episode_length=800, batch_size=128, num_minibatches=4, unroll_length=20,
                    num_updates_per_batch=4, num_resets_per_eval=0, learning_rate=args.learning_rate,
                    entropy_cost=.001, discounting=.99, normalize_observations=False, reward_scaling=1,
                    network_factory=network_factory,
                    wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
                    seed=args.seed, num_evals=args.evals, run_evals=False, restore_params=params,
                    restore_value_fn=True, policy_params_fn=callback, progress_fn=progress)
    best = max(rows, key=lambda row: row["success30"])
    summary = dict(first=rows[0]["success30"], last=rows[-1]["success30"], max=best["success30"],
                   best_step=best["step"], rows=len(rows), final_step=rows[-1]["step"],
                   elapsed_sec=time.monotonic()-started, bc_elapsed_sec=meta["bc_elapsed_sec"],
                   status="completed")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
