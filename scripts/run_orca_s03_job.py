"""Run one S03 job; training and substep-audited evaluation use separate envs."""

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

from brax.training.agents.ppo import train
import jax
import numpy as np
from mujoco_playground._src import wrapper

from orca_s03_env import OrcaS03Grasp
from prepare_hic_orca_object import digest
from run_orca_suite_job import MODEL, REFERENCE, DATASET, evaluator, summarize
from train_orca_acquisition import initial_gate
from train_orca_comparison import network_factory
from train_orca_matched import initialize_matched, dataset_arrays, fit_matched_bc


def validate_job(job):
    if job["method"] not in ("ppo", "guided", "residual", "zero", "bc"):
        raise ValueError("Unknown method")
    if job["steps"] < 0 or job["eval_count"] < 1:
        raise ValueError("Invalid budget/population")
    if job["steps"] and (job.get("checkpoint") or job["method"] in ("zero", "bc")):
        raise ValueError("Training starts from matched initialization, not old weights")
    if job["steps"] and job.get("benchmark", "B0_nominal") != "B0_nominal":
        raise ValueError("Only nominal-task training is allowed")
    if job["steps"] and job["eval_start"] >= 71000:
        raise ValueError("Training may not inspect final test seeds")
    if job["steps"] and (job["steps"] % 10240 or job["evals"] < 2):
        raise ValueError("Training must use whole rollout batches and >=2 checkpoints")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    validate_job(job)
    out = ROOT / job["output"]
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    jax.config.update("jax_pmap_shmap_merge", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "artifacts/orca30/jax_cache"))
    if jax.default_backend() != "gpu":
        raise RuntimeError("S03 requires the verified aero_rl GPU backend")
    arm = "residual" if job["method"] in ("residual", "zero") else "pure"
    settings = dict(benchmark=job.get("benchmark", "B0_nominal"),
                    shaping=job.get("shaping", 1.), push_force=job.get("push_force"),
                    residual_gain=0. if job["method"] == "zero" else 1.)
    env = OrcaS03Grasp(MODEL, arm, REFERENCE, audit=False, **settings)
    eval_env = OrcaS03Grasp(MODEL, arm, REFERENCE, audit=True, **settings)
    gate = initial_gate(eval_env)
    net, params = initialize_matched(env, job["seed"], job["std"])
    meta = dict(job, job_sha256=digest(args.job), model=str(MODEL), reference=str(REFERENCE),
                model_sha256=digest(MODEL), reference_sha256=digest(REFERENCE),
                dataset_sha256=digest(DATASET) if job["method"] in ("guided", "bc") else None,
                arm=arm, actor_dim=123, critic_dim=140, initial_gate=gate,
                physics_backend=env.physics_backend, evaluation_native_substep_audit=True,
                bc_elapsed_sec=0., xla_flags=os.environ.get("XLA_FLAGS", ""), source_sha256={})
    sources = out / "sources"
    sources.mkdir()
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().parent == ROOT / "scripts" and filename.endswith(".py"):
            path = Path(filename).resolve()
            shutil.copyfile(path, sources / path.name)
            meta["source_sha256"][path.name] = digest(path)
    if job.get("checkpoint"):
        path = ROOT / job["checkpoint"]
        config = path.parent / "run_config.json"
        if not config.exists():
            config = path.parent.parent / "run_config.json"
        prior = json.loads(config.read_text())
        for key in ("model_sha256", "reference_sha256", "arm", "actor_dim", "critic_dim"):
            if meta[key] != prior[key]:
                raise ValueError("Checkpoint task mismatch: " + key)
        params = pickle.loads(path.read_bytes())
        meta["checkpoint_sha256"] = digest(path)
    elif job["method"] in ("guided", "bc"):
        arrays = dataset_arrays(DATASET, True, digest(MODEL), digest(REFERENCE), env.physics_backend)
        before = time.monotonic()
        params, losses = fit_matched_bc(net, params, arrays, 4000, job["seed"])
        meta["bc_elapsed_sec"] = time.monotonic() - before
        (out / "bc_loss.json").write_text(json.dumps(losses, indent=2)+"\n")
        (out / "bc_params.pkl").write_bytes(pickle.dumps(jax.device_get(params)))
    (out / "run_config.json").write_text(json.dumps(meta, indent=2)+"\n")
    (out / "env_config.json").write_text(env._config.to_json(indent=2))
    seeds = list(range(job["eval_start"], job["eval_start"] + job["eval_count"]))
    evaluate = evaluator(eval_env, net, seeds)
    rows = []

    def callback(step, _policy, current):
        canonical = jax.tree_util.tree_map(np.asarray, jax.device_get(current))
        initial, trace = jax.device_get(evaluate(canonical))
        if not np.isfinite(initial).all() or not all(np.isfinite(v).all() for v in trace.values()):
            raise FloatingPointError("Nonfinite evaluation")
        row, episodes = summarize(trace, eval_env, seeds)
        row.update(step=int(step), elapsed_sec=time.monotonic()-started)
        directory = out / f"step_{int(step):09d}"
        directory.mkdir(exist_ok=False)
        (directory / "episodes.json").write_text(json.dumps(episodes, indent=2)+"\n")
        np.savez_compressed(directory / "initial_states.npz", qpos=initial, seeds=seeds)
        np.savez_compressed(directory / "preview_trajectories.npz", **{k: v[:4] for k, v in trace.items()})
        blob = pickle.dumps(canonical)
        (out / "params.pkl").write_bytes(blob)
        (directory / "params.pkl").write_bytes(blob)
        rows.append(row)
        with (out / "metrics.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)

    def progress(step, metrics):
        values = {k: float(v) for k, v in metrics.items()}
        if not np.isfinite(list(values.values())).all():
            raise FloatingPointError("Nonfinite PPO update")
        with (out / "ppo_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(step=int(step), **values))+"\n")

    if job["steps"]:
        train.train(environment=env, num_timesteps=job["steps"], num_envs=128,
                    num_eval_envs=len(seeds), episode_length=800, batch_size=128,
                    num_minibatches=4, unroll_length=20, num_updates_per_batch=4,
                    num_resets_per_eval=0, learning_rate=job["learning_rate"],
                    entropy_cost=.001, discounting=.99, normalize_observations=False,
                    reward_scaling=1, network_factory=network_factory,
                    wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
                    seed=job["seed"], num_evals=job["evals"], run_evals=False, restore_params=params,
                    restore_value_fn=True, policy_params_fn=callback, progress_fn=progress)
    else:
        callback(0, None, params)
    best = max(rows, key=lambda row: row["clean_success30"])
    steps = np.array([r["step"] for r in rows])
    clean = np.array([r["clean_success30"] for r in rows])
    area = float(np.trapezoid(clean, steps) / steps[-1]) if steps[-1] else float(clean[0])
    summary = dict(first=float(clean[0]), last=float(clean[-1]), max=float(clean.max()),
                   best_step=best["step"], final_step=int(steps[-1]), clean_auc=area,
                   raw_first=rows[0]["success30"], raw_last=rows[-1]["success30"],
                   final_force_n=rows[-1]["mean_finger_peak_n"],
                   elapsed_sec=time.monotonic()-started, bc_elapsed_sec=meta["bc_elapsed_sec"],
                   status="completed", job_sha256=meta["job_sha256"])
    if job["steps"] and steps[-1] != job["steps"]:
        raise RuntimeError("Actual training step count differs from budget")
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
