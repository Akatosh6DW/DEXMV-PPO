"""One reproducible cylinder job: collect, BC, PPO, or independent evaluation."""

import argparse
import csv
import functools
import importlib.metadata
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

from orca_cylinder_evaluation import evaluator
from orca_shape_pilot_env import OrcaShapePilot
from orca_residual_controls import load_reference
from prepare_hic_orca_object import digest
from run_orca_suite_job import MODEL, summarize
from train_orca_acquisition import initial_gate
from train_orca_comparison import network_factory
from train_orca_matched import initialize_matched, dataset_arrays, fit_matched_bc


def validate(job):
    if job["method"] not in ("ppo", "guided", "residual", "zero", "bc", "collect"):
        raise ValueError("Unknown method")
    if job["steps"] < 0 or job["steps"] % 10240 or job["eval_count"] < 1:
        raise ValueError("Invalid training budget or population")
    if job["steps"] and (job["method"] in ("zero", "bc", "collect") or job.get("checkpoint")):
        raise ValueError("Formal training must not continue selected old weights")
    if job["steps"] and job.get("benchmark", "B0_nominal") != "B0_nominal":
        raise ValueError("Robustness conditions are evaluation-only")
    if job["steps"] and job["eval_start"] >= 110000:
        raise ValueError("Training cannot inspect final test seeds")
    if job["steps"] and (job.get("evals", 5) < 2 or
                         job["steps"] % ((job.get("evals", 5)-1)*10240)):
        raise ValueError("Evaluation intervals must use whole rollout batches")
    if job["method"] in ("guided", "bc") and not (job.get("dataset") or job.get("checkpoint")):
        raise ValueError("BC requires an explicitly bound demonstration dataset")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    validate(job)
    out = ROOT / job["output"]
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    jax.config.update("jax_pmap_shmap_merge", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "artifacts/orca30/jax_cache"))
    reference = ROOT / job["reference"]
    arm = "residual" if job["method"] in ("zero", "residual") else "pure"
    settings = dict(shape="cylinder", shaping=job.get("shaping", 1.),
                    benchmark=job.get("benchmark", "B0_nominal"),
                    residual_gain=0. if job["method"] == "zero" else 1.)
    env = OrcaShapePilot(MODEL, arm, reference, audit=False, **settings)
    eval_env = OrcaShapePilot(MODEL, arm, reference, audit=True, **settings)
    gate = initial_gate(eval_env)
    (out / "model.xml").write_text(eval_env.evaluation_xml+"\n")
    model_hash = digest(out / "model.xml")
    net, params = initialize_matched(env, job["seed"], job.get("std", .7))
    meta = dict(job, arm=arm, actor_dim=123, critic_dim=140, initial_gate=gate,
                model=str((out / "model.xml").resolve()), model_sha256=model_hash,
                reference_sha256=digest(reference), source_model_sha256=digest(MODEL),
                job_sha256=digest(args.job), physics_backend=env.physics_backend,
                backend=jax.default_backend(), command=sys.argv, source_sha256={}, bc_elapsed_sec=0.,
                versions={name:importlib.metadata.version(name) for name in ("jax", "jaxlib", "mujoco", "brax", "flax", "optax")},
                extra_cost="Reference simulation search and BC are additional to equal online PPO steps",
                xla_flags=os.environ.get("XLA_FLAGS", ""))
    sources = out / "sources"
    sources.mkdir()
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().parent == ROOT / "scripts" and filename.endswith(".py"):
            path = Path(filename).resolve()
            shutil.copyfile(path, sources / path.name)
            meta["source_sha256"][path.name] = digest(path)
    if job.get("checkpoint"):
        checkpoint = ROOT / job["checkpoint"]
        config = checkpoint.parent / "run_config.json"
        if not config.exists():
            config = checkpoint.parent.parent / "run_config.json"
        previous = json.loads(config.read_text())
        for key in ("arm", "actor_dim", "critic_dim", "model_sha256", "reference_sha256"):
            if previous[key] != meta[key]:
                raise ValueError("Incompatible checkpoint " + key)
        params = pickle.loads(checkpoint.read_bytes())
        meta["checkpoint_sha256"] = digest(checkpoint)
    elif job["method"] in ("guided", "bc"):
        path = ROOT / job["dataset"]
        arrays = dataset_arrays(path, True, model_hash, digest(reference), env.physics_backend)
        before = time.monotonic()
        params, losses = fit_matched_bc(net, params, arrays, job.get("bc_updates", 4000), job["seed"])
        meta.update(dataset_sha256=digest(path), bc_elapsed_sec=time.monotonic()-before)
        (out / "bc_loss.json").write_text(json.dumps(losses, indent=2)+"\n")
        (out / "bc_params.pkl").write_bytes(pickle.dumps(jax.device_get(params)))
    (out / "run_config.json").write_text(json.dumps(meta, indent=2)+"\n")
    (out / "env_config.json").write_text(env._config.to_json(indent=2))
    shutil.copyfile(reference, out / "reference.npz")
    teacher = None
    if job["method"] == "collect":
        teacher, _ = load_reference(reference, env._lowers, env._uppers,
                                    env._config.initial_ctrl, .05, 800, .1)
    seeds = list(range(job["eval_start"], job["eval_start"]+job["eval_count"]))
    evaluate = evaluator(eval_env, net, seeds, teacher)
    rows = []

    def callback(step, _policy, current):
        canonical = jax.tree_util.tree_map(np.asarray, jax.device_get(current))
        initial, trace = jax.device_get(evaluate(canonical))
        if not np.isfinite(initial).all() or not all(np.isfinite(v).all() for v in trace.values()):
            raise FloatingPointError("Nonfinite rollout")
        row, episodes = summarize(trace, eval_env, seeds)
        for i, ep in enumerate(episodes):
            ep["whole_episode_clean_success30"] = bool(ep["clean_success30"] and ep["max_penetration_m"] <= .003)
            ep["drop"] = bool(not trace["alive"][i, -1])
        row["whole_episode_clean_success30"] = float(np.mean([e["whole_episode_clean_success30"] for e in episodes]))
        row.update(step=int(step), elapsed_sec=time.monotonic()-started)
        if arm == "pure":
            np.testing.assert_array_equal(trace["video_base"], 0.)
        if job["method"] == "zero":
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
        if job["method"] == "collect":
            selected = np.array([e["whole_episode_clean_success30"] for e in episodes])
            if selected.sum() < 4:
                raise ValueError("Fewer than four clean demonstrations")
            np.testing.assert_array_equal(trace["obs"][..., 95:112], 0.)
            target = out / "trajectories.npz"
            np.savez_compressed(target, **{k:trace[k][selected] for k in ("obs", "action", "active")})
            manifest = dict(dataset_sha256=digest(target), model_sha256=model_hash,
                            reference_sha256=digest(reference), actor_dim=123,
                            physics_backend=env.physics_backend, seeds=np.array(seeds)[selected].tolist(),
                            attempted=len(seeds), accepted=int(selected.sum()),
                            source_video_count=1, source_object="cube", target_object="cylinder",
                            teacher="Video-derived trajectory plus explicit simulator calibration",
                            split="First 75% independent rollouts train; remaining rollouts validate")
            (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")

    def progress(step, metrics):
        values = {k:float(v) for k,v in metrics.items()}
        if not np.isfinite(list(values.values())).all():
            raise FloatingPointError("Nonfinite PPO update")
        with (out / "ppo_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(step=int(step), **values))+"\n")

    if job["steps"]:
        train.train(environment=env, num_timesteps=job["steps"], num_envs=128,
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
    if rows[-1]["step"] != job["steps"]:
        raise ValueError("Incomplete training budget")
    best = max(rows, key=lambda r:r["clean_success30"])
    summary = dict(first=rows[0]["clean_success30"], last=rows[-1]["clean_success30"],
                   max=best["clean_success30"], best_step=best["step"], actual_steps=rows[-1]["step"],
                   job_sha256=digest(args.job), elapsed_sec=time.monotonic()-started, status="completed")
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
