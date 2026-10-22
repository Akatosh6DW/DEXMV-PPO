"""Small shape-transfer evaluation or bounded PPO continuation, not a benchmark."""

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
import numpy as np
from mujoco_playground._src import wrapper

from orca_shape_pilot_env import OrcaShapePilot, SHAPES
from prepare_hic_orca_object import digest
from run_orca_suite_job import MODEL, REFERENCE, evaluator, summarize
from train_orca_acquisition import initial_gate
from train_orca_comparison import network_factory
from train_orca_matched import initialize_matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=SHAPES, required=True)
    parser.add_argument("--method", choices=("zero", "residual"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--push-force", type=float, default=0.)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--evals", type=int, default=3)
    parser.add_argument("--eval-count", type=int, default=8)
    parser.add_argument("--eval-start", type=int, default=92000)
    parser.add_argument("--seed", type=int, default=31001)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--height-weight", type=float, default=0.)
    args = parser.parse_args()
    if args.method == "zero" and (args.steps or args.checkpoint):
        parser.error("Reference-only evaluations cannot load weights or train")
    if args.steps < 0 or args.steps % 10240 or args.eval_count < 1 or (args.steps and args.evals < 2):
        parser.error("Use whole 10240-step batches, positive episode count and >=2 training evaluations")
    args.output.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_pmap_shmap_merge", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "artifacts/orca30/jax_cache"))
    started = time.monotonic()
    settings = dict(shape=args.shape, push_force=args.push_force,
                    residual_gain=0. if args.method == "zero" else 1., shaping=0.,
                    height_weight=args.height_weight)
    env = OrcaShapePilot(MODEL, "residual", args.reference, audit=False, **settings)
    eval_env = OrcaShapePilot(MODEL, "residual", args.reference, audit=True, **settings)
    gate = initial_gate(eval_env, 16)
    net, params = initialize_matched(env, args.seed, .7)
    meta = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    meta.update(development=True, backend=jax.default_backend(), initial_gate=gate,
                actor_dim=123, critic_dim=140, arm="residual", source_model_sha256=digest(MODEL),
                reference_sha256=digest(args.reference), source_sha256={},
                reference_note="Cube-video reference transferred to another shape; no new video extraction",
                checkpoint_note="PPO continuation restores networks, not the previous optimizer state")
    if args.checkpoint:
        config = args.checkpoint.parent / "run_config.json"
        if not config.exists():
            config = args.checkpoint.parent.parent / "run_config.json"
        prior = json.loads(config.read_text())
        for key in ("actor_dim", "critic_dim", "arm", "reference_sha256"):
            if prior[key] != meta[key]:
                raise ValueError("Incompatible checkpoint: " + key)
        if prior.get("source_model_sha256", prior.get("model_sha256")) != digest(MODEL):
            raise ValueError("Unexpected source hand model")
        params = pickle.loads(args.checkpoint.read_bytes())
        meta["checkpoint_sha256"] = digest(args.checkpoint)
    model_path = args.output / "model.xml"
    model_path.write_text(eval_env.evaluation_xml + "\n")
    meta.update(model=str(model_path.resolve()), model_sha256=digest(model_path))
    sources = args.output / "sources"
    sources.mkdir()
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().parent == ROOT / "scripts" and filename.endswith(".py"):
            path = Path(filename).resolve()
            shutil.copyfile(path, sources / path.name)
            meta["source_sha256"][path.name] = digest(path)
    (args.output / "run_config.json").write_text(json.dumps(meta, indent=2)+"\n")
    (args.output / "env_config.json").write_text(env._config.to_json(indent=2))
    shutil.copyfile(args.reference, args.output / "reference.npz")
    seeds = list(range(args.eval_start, args.eval_start + args.eval_count))
    evaluate = evaluator(eval_env, net, seeds)
    rows = []

    def callback(step, _policy, current):
        canonical = jax.tree_util.tree_map(np.asarray, jax.device_get(current))
        initial, trace = jax.device_get(evaluate(canonical))
        if not np.isfinite(initial).all() or not all(np.isfinite(v).all() for v in trace.values()):
            raise FloatingPointError("Nonfinite rollout")
        row, episodes = summarize(trace, eval_env, seeds)
        if args.method == "zero":
            np.testing.assert_allclose(trace["ctrl"], trace["video_base"], atol=1e-5)
        if args.push_force:
            np.testing.assert_allclose(np.linalg.norm(trace["push"][:, 200:204], axis=-1),
                                       args.push_force, atol=1e-6)
        row.update(step=int(step), elapsed_sec=time.monotonic()-started)
        directory = args.output / f"step_{int(step):09d}"
        directory.mkdir(exist_ok=False)
        (directory / "episodes.json").write_text(json.dumps(episodes, indent=2)+"\n")
        np.savez_compressed(directory / "initial_states.npz", qpos=initial, seeds=seeds)
        np.savez_compressed(directory / "preview_trajectories.npz", **{k: v[:4] for k, v in trace.items()})
        (directory / "params.pkl").write_bytes(pickle.dumps(canonical))
        rows.append(row)
        with (args.output / "metrics.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)

    def progress(step, metrics):
        values = {k: float(v) for k, v in metrics.items()}
        if not np.isfinite(list(values.values())).all():
            raise FloatingPointError("Nonfinite PPO update")
        with (args.output / "ppo_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(step=int(step), **values))+"\n")

    if args.steps:
        train.train(environment=env, num_timesteps=args.steps, num_envs=128,
                    num_eval_envs=len(seeds), episode_length=800, batch_size=128,
                    num_minibatches=4, unroll_length=20, num_updates_per_batch=4,
                    num_resets_per_eval=0, learning_rate=args.learning_rate,
                    entropy_cost=.001, discounting=.99, normalize_observations=False,
                    reward_scaling=1, network_factory=network_factory,
                    wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
                    seed=args.seed, num_evals=args.evals, run_evals=False, restore_params=params,
                    restore_value_fn=True, policy_params_fn=callback, progress_fn=progress)
    else:
        callback(0, None, params)
    best = max(rows, key=lambda r: r["clean_success30"])
    summary = dict(first=rows[0]["clean_success30"], last=rows[-1]["clean_success30"],
                   max=best["clean_success30"], best_step=best["step"], actual_steps=rows[-1]["step"],
                   elapsed_sec=time.monotonic()-started, status="completed")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
