"""One fresh raw-video cylinder job with explicit inputs and no warm starts."""

import argparse
import csv
import functools
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts/orca_cylinder_raw_video_20261009"
sys.path.insert(0, str(ROOT / "Aero-Hand/sim_rl/mujoco_playground"))

from brax.training.agents.ppo import train
import jax
import numpy as np
from mujoco_playground._src import wrapper

from orca_raw_can_env import OrcaRawCanGrasp
from orca_raw_can_bc import raw_dataset_arrays
from orca_cylinder_evaluation import evaluator
from orca_residual_controls import load_reference
from run_orca_suite_job import summarize
from train_orca_acquisition import initial_gate
from train_orca_comparison import network_factory
from train_orca_matched import initialize_matched, fit_matched_bc


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fresh_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(STUDY.resolve()):
        raise ValueError("Old task data/checkpoints are not permitted: " + str(path))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    method = job["method"]
    if method not in ("ppo", "guided", "residual", "bc", "zero", "collect"):
        raise ValueError("Unknown method")
    steps = job["steps"]
    if steps < 0 or steps % 10240 or (steps and (job.get("checkpoint") or method not in ("ppo", "guided", "residual"))):
        raise ValueError("Invalid budget or warm start")
    if steps and job["eval_start"] >= 210000:
        raise ValueError("No final test seeds during training")
    if steps and (job.get("evals", 5) < 2 or steps % ((job.get("evals", 5)-1)*10240)):
        raise ValueError("Incomplete rollout/evaluation intervals")
    out = fresh_path(job["output"])
    out.mkdir(parents=True, exist_ok=False)
    model, reference = fresh_path(job["model"]), fresh_path(job["reference"])
    jax.config.update("jax_pmap_shmap_merge", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", str(STUDY / "jax_cache"))
    started = time.monotonic()
    arm = "residual" if method in ("residual", "zero") else "pure"
    settings = dict(benchmark=job.get("benchmark", "B0_nominal"), residual_gain=0. if method == "zero" else 1.)
    env = OrcaRawCanGrasp(model, arm, reference, audit=False, **settings)
    eval_env = OrcaRawCanGrasp(model, arm, reference, audit=True, **settings)
    gate = initial_gate(eval_env)
    net, params = initialize_matched(env, job["seed"], job.get("std", .3))
    meta = dict(job, arm=arm, actor_dim=123, critic_dim=140, initial_gate=gate,
                model_sha256=digest(model), reference_sha256=digest(reference),
                job_sha256=digest(args.job), physics_backend=env.physics_backend,
                backend=jax.default_backend(), command=sys.argv, source_sha256={},
                bc_elapsed_sec=0., initialization="fresh PRNG seed; no task checkpoint loaded",
                extra_cost="Fresh RGB inference, geometric retargeting and replay dataset; no success-based calibration",
                xla_flags=os.environ.get("XLA_FLAGS", ""))
    if job.get("checkpoint"):
        checkpoint = fresh_path(job["checkpoint"])
        config = checkpoint.parent / "run_config.json"
        if not config.exists():
            config = checkpoint.parent.parent / "run_config.json"
        previous = json.loads(config.read_text())
        for key in ("arm", "actor_dim", "critic_dim", "model_sha256", "reference_sha256"):
            if previous[key] != meta[key]:
                raise ValueError("Checkpoint mismatch: " + key)
        params = pickle.loads(checkpoint.read_bytes())
        meta.update(checkpoint_sha256=digest(checkpoint), initialization="evaluation of new-study checkpoint only")
    elif method in ("guided", "bc"):
        path = fresh_path(job["dataset"])
        arrays = raw_dataset_arrays(path, digest(model), digest(reference), env.physics_backend)
        before = time.monotonic()
        params, losses = fit_matched_bc(net, params, arrays, 4000, job["seed"])
        meta.update(dataset_sha256=digest(path), bc_elapsed_sec=time.monotonic()-before,
                    bc_sampling="Uniform active replay samples; no successful-hold phase assumed")
        (out / "bc_loss.json").write_text(json.dumps(losses, indent=2)+"\n")
        (out / "bc_params.pkl").write_bytes(pickle.dumps(jax.device_get(params)))
    sources = out / "sources"
    sources.mkdir()
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().parent == ROOT / "scripts" and filename.endswith(".py"):
            path = Path(filename).resolve()
            shutil.copyfile(path, sources / path.name)
            meta["source_sha256"][path.name] = digest(path)
    (out / "run_config.json").write_text(json.dumps(meta, indent=2)+"\n")
    (out / "env_config.json").write_text(env._config.to_json(indent=2))
    shutil.copyfile(reference, out / "reference.npz")
    shutil.copyfile(model, out / "model.xml")
    teacher = None
    if method == "collect":
        teacher, _ = load_reference(reference, env._lowers, env._uppers, env._config.initial_ctrl, .05, 800, .1)
    seeds = list(range(job["eval_start"], job["eval_start"]+job["eval_count"]))
    evaluate = evaluator(eval_env, net, seeds, teacher)
    rows = []

    def callback(step, _policy, current):
        canonical = jax.tree_util.tree_map(np.asarray, jax.device_get(current))
        initial, trace = jax.device_get(evaluate(canonical))
        if not np.isfinite(initial).all() or not all(np.isfinite(v).all() for v in trace.values()):
            raise FloatingPointError("Nonfinite evaluation")
        row, episodes = summarize(trace, eval_env, seeds)
        for i, ep in enumerate(episodes):
            ep["drop"] = bool(not trace["alive"][i, -1])
            ep["whole_episode_clean_success30"] = bool(ep["clean_success30"] and ep["max_penetration_m"] <= .003)
        row.update(step=int(step), elapsed_sec=time.monotonic()-started)
        if arm == "pure":
            np.testing.assert_array_equal(trace["video_base"], 0.)
        if method == "zero":
            np.testing.assert_allclose(trace["ctrl"], trace["video_base"], atol=1e-5)
        folder = out / f"step_{int(step):09d}"
        folder.mkdir(exist_ok=False)
        (folder / "episodes.json").write_text(json.dumps(episodes, indent=2)+"\n")
        np.savez_compressed(folder / "initial_states.npz", qpos=initial, seeds=seeds)
        np.savez_compressed(folder / "preview_trajectories.npz", **{k:v[:4] for k,v in trace.items()})
        (folder / "params.pkl").write_bytes(pickle.dumps(canonical))
        rows.append(row)
        with (out / "metrics.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)
        if method == "collect":
            # Keep every replay, including failures. This is video imitation data,
            # not a claim that the robot replay is a successful expert rollout.
            np.testing.assert_array_equal(trace["obs"][..., 95:112], 0.)
            target = out / "trajectories.npz"
            np.savez_compressed(target, **{k:trace[k] for k in ("obs", "action", "active")})
            manifest = dict(dataset_sha256=digest(target), model_sha256=digest(model),
                            reference_sha256=digest(reference), actor_dim=123,
                            physics_backend=env.physics_backend, seeds=seeds,
                            attempted=len(seeds), retained=len(seeds),
                            clean_success_count=sum(e["clean_success30"] for e in episodes),
                            source_video_count=1, source_object="tomato_soup_can", target_object="cylinder",
                            teacher="Fresh RGB hand inference and geometric retargeting, no success selection",
                            split="First 75% rollouts train; remaining rollouts validation; one source video shared")
            (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")

    def progress(step, metrics):
        values = {k:float(v) for k,v in metrics.items()}
        if not np.isfinite(list(values.values())).all():
            raise FloatingPointError("Nonfinite PPO update")
        with (out / "ppo_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(step=int(step), **values))+"\n")

    if steps:
        train.train(environment=env, num_timesteps=steps, num_envs=128,
                    num_eval_envs=len(seeds), episode_length=800, batch_size=128,
                    num_minibatches=4, unroll_length=20, num_updates_per_batch=4,
                    num_resets_per_eval=0, learning_rate=job["learning_rate"], entropy_cost=.001,
                    discounting=.99, normalize_observations=False, reward_scaling=1,
                    network_factory=network_factory,
                    wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
                    seed=job["seed"], num_evals=job.get("evals", 5), run_evals=False,
                    restore_params=params, restore_value_fn=True, policy_params_fn=callback, progress_fn=progress)
    else:
        callback(0, None, params)
    if rows[-1]["step"] != steps:
        raise ValueError("Incomplete training budget")
    best = max(rows, key=lambda row:row["clean_success30"])
    summary = dict(first=rows[0]["clean_success30"], last=rows[-1]["clean_success30"],
                   max=best["clean_success30"], best_step=best["step"], actual_steps=rows[-1]["step"],
                   job_sha256=digest(args.job), elapsed_sec=time.monotonic()-started, status="completed")
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
