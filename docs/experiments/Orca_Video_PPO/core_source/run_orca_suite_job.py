# 中文阅读副本：16 S02训练与正式评分
# 原始入口：scripts/run_orca_suite_job.py
# 仅用于阅读，保留原路径假设；不要从此副本启动训练。运行请使用原始入口。
# 本副本只增加注释，生成时核验AST与非注释词法序列完全一致。
"""One frozen S02 training/evaluation job, with independent episode metrics."""

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
import mujoco
import numpy as np
from mujoco_playground._src import wrapper

from orca_suite_env import OrcaSuiteGrasp
from prepare_hic_orca_object import digest
from train_orca_acquisition import MODEL, REFERENCE, episode_scores, initial_gate
from train_orca_comparison import network_factory
from train_orca_matched import initialize_matched, dataset_arrays, fit_matched_bc

RAW_REFERENCE = MODEL.parent / "reference.npz"
DATASET = ROOT / "artifacts/orca_cube_acquisition/D01_quality_demos/trajectories.npz"


def evaluator(env, net, seeds):
    inference = networks.make_inference_fn(net)
    def evaluate(params):
        policy = inference(params, deterministic=True)
        def episode(key):
            state = env.reset(key)
            start_qpos = state.data.qpos
            def step(carry, _):
                state, alive, key = carry
                key, subkey = jax.random.split(key)
                action = policy(state.obs, subkey)[0]
                nxt = env.step(state, action)
                next_alive = alive & (nxt.done == 0)
                m = nxt.metrics
                executed = action * env.residual_gain if env.arm == "residual" else action
                trace = dict(action=executed, policy_action=action, active=alive, alive=next_alive,
                             hold=jp.where(next_alive, m["hold_sec"], 0),
                             reward=jp.where(alive, nxt.reward, 0),
                             qpos=nxt.data.qpos, qvel=nxt.data.qvel, ctrl=nxt.data.ctrl,
                             mocap=nxt.data.mocap_pos, forbidden=m["forbidden_contact"],
                             contacts=m["contact_count"], thumb=m["thumb_contact"],
                             released=m["released"], cube_z=m["cube_z"],
                             spawn_z=nxt.info["cube_spawn_pos"][2], cube_speed=m["cube_speed"],
                             finger_normal_force=nxt.info["finger_normal_force"],
                             requested_residual=nxt.info["requested_residual"],
                             effective_residual=nxt.info["effective_residual"],
                             desired_ctrl=nxt.info["desired_ctrl"],
                             video_base=nxt.info["video_base"], push=nxt.info["applied_push"],
                             substep_penetration=m["substep_penetration"],
                             substep_forbidden=m["substep_forbidden"],
                             substep_force_peak=m["substep_force_peak"])
                return (nxt, next_alive, key), trace
            trace = jax.lax.scan(step, (state, jp.bool_(True), jax.random.fold_in(key, 987654)),
                                 None, length=800)[1]
            return start_qpos, trace
        return jax.vmap(episode)(jax.vmap(jax.random.PRNGKey)(jp.asarray(seeds, jp.int32)))
    return jax.jit(evaluate)


# 导读：先算连续保持，再用撤托后的每5ms接触审查判定清洁成功；高reward不等于成功。
def summarize(trace, env, seeds):
    row, episodes = episode_scores(trace, env, seeds)
    for i, episode in enumerate(episodes):
        active = trace["active"][i]
        released = active & (trace["released"][i] > 0)
        depths = trace["substep_penetration"][i]
        forbidden = bool(np.any((trace["substep_forbidden"][i] > 0) & released))
        depth = float(np.max(depths[released], initial=0.))
        episode.update(max_penetration_m=float(np.max(depths[active], initial=0.)),
                       post_release_penetration_m=depth, substep_forbidden=forbidden,
                       substep_force_peak_n=float(np.max(trace["substep_force_peak"][i][active], initial=0.)),
                       clean_success30=bool(episode["success30"] and not forbidden and depth <= .003),
                       residual_rms_rad=float(np.sqrt(np.mean(trace["requested_residual"][i][active]**2))),
                       valid_before_push=bool(trace["alive"][i, 199] and trace["hold"][i, 199] > 0),
                       valid_one_second_after_push=bool(trace["alive"][i, 223] and trace["hold"][i, 223] > 0))
    row.update(clean_success30=float(np.mean([e["clean_success30"] for e in episodes])),
               max_penetration_m=max(e["max_penetration_m"] for e in episodes),
               substep_forbidden_rate=float(np.mean([e["substep_forbidden"] for e in episodes])))
    return row, episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    out = ROOT / job["output"]
    out.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_pmap_shmap_merge", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "artifacts/orca30/jax_cache"))
    assert jax.default_backend() == "gpu"
    started = time.monotonic()
    method = job["method"]
    arm = "residual" if method in ("residual", "linear", "unrefined", "zero") else "pure"
    variant = job.get("variant", method if method in ("linear", "unrefined") else "video")
    env = OrcaSuiteGrasp(MODEL, arm, REFERENCE, benchmark=job.get("benchmark", "B0_nominal"),
                        variant=variant, raw_reference=RAW_REFERENCE,
                        audit=job.get("audit", False), residual_gain=job.get("residual_gain", 1.))
    gate = initial_gate(env, 16)
    seed, std = job["seed"], job.get("std", .3)
    net, params = initialize_matched(env, seed, std)
    meta = dict(job, job_sha256=digest(args.job), model=str(MODEL), reference=str(REFERENCE),
                model_sha256=digest(MODEL), reference_sha256=digest(REFERENCE),
                raw_reference_sha256=digest(RAW_REFERENCE), dataset_sha256=digest(DATASET),
                variant=variant, arm=arm, actor_dim=123, critic_dim=140, initial_gate=gate,
                physics_backend=env.physics_backend, jax_version=jax.__version__,
                mujoco_version=mujoco.__version__, bc_elapsed_sec=0.,
                xla_flags=os.environ.get("XLA_FLAGS", ""), source_sha256={})
    sources = out / "sources"
    sources.mkdir()
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name and Path(name).resolve().parent == ROOT / "scripts" and name.endswith('.py'):
            path = Path(name).resolve()
            shutil.copyfile(path, sources / path.name)
            meta["source_sha256"][path.name] = digest(path)
    if job.get("checkpoint"):
        path = ROOT / job["checkpoint"]
        prior = json.loads((path.parent / "run_config.json").read_text())
        for name in ("model_sha256", "reference_sha256", "raw_reference_sha256", "actor_dim", "critic_dim", "variant", "arm"):
            if prior[name] != meta[name]:
                raise ValueError("Checkpoint mismatch: " + name)
        params = pickle.loads(path.read_bytes())
        meta["checkpoint_sha256"] = digest(path)
    # 导读：guided和仅BC共享4000步BC初始化；guided随后调用PPO，bc在初始化后停止。
    elif method in ("guided", "bc"):
        arrays = dataset_arrays(DATASET, True, digest(MODEL), digest(REFERENCE), env.physics_backend)
        before = time.monotonic()
        params, losses = fit_matched_bc(net, params, arrays, 4000, seed)
        meta["bc_elapsed_sec"] = time.monotonic() - before
        (out / "bc_loss.json").write_text(json.dumps(losses, indent=2))
        (out / "bc_params.pkl").write_bytes(pickle.dumps(jax.device_get(params)))
    (out / "run_config.json").write_text(json.dumps(meta, indent=2)+"\n")
    (out / "env_config.json").write_text(env._config.to_json(indent=2))
    seeds = list(range(job["eval_start"], job["eval_start"] + job["eval_count"]))
    evaluate = evaluator(env, net, seeds)
    rows = []

    def callback(step, _policy, current):
        canonical = jax.tree_util.tree_map(np.asarray, jax.device_get(current))
        initial, trace = jax.device_get(evaluate(canonical))
        assert np.isfinite(initial).all() and all(np.isfinite(v).all() for v in trace.values())
        row, episodes = summarize(trace, env, seeds)
        if not job.get("audit", False):
            row["clean_success30"] = None
            row["max_penetration_m"] = None
            row["substep_forbidden_rate"] = None
            for episode in episodes:
                for name in ("clean_success30", "max_penetration_m", "post_release_penetration_m",
                             "substep_forbidden", "substep_force_peak_n"):
                    episode[name] = None
        row.update(step=int(step), elapsed_sec=time.monotonic()-started)
        rows.append(row)
        directory = out / f"step_{int(step):09d}"
        directory.mkdir(exist_ok=False)
        (directory / "episodes.json").write_text(json.dumps(episodes, indent=2)+"\n")
        np.savez_compressed(directory / "initial_states.npz", qpos=initial, seeds=seeds)
        keep = min(4, len(seeds))
        np.savez_compressed(directory / "preview_trajectories.npz", **{k: v[:keep] for k, v in trace.items()})
        (out / "params.pkl").write_bytes(pickle.dumps(canonical))
        with (out / "metrics.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)

    def progress(step, metrics):
        values = {k: float(v) for k, v in metrics.items()}
        assert np.isfinite(list(values.values())).all()
        with (out / "ppo_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(step=int(step), **values))+"\n")

    if job.get("steps", 0):
        if job.get("checkpoint") or job.get("audit") or method in ("zero", "bc"):
            raise ValueError("Invalid training job")
        # 导读：调用Brax现成PPO；本项目负责环境、先验接口、初始化及评估，不自行另写PPO公式。
        train.train(environment=env, num_timesteps=job["steps"], num_envs=128,
                    num_eval_envs=len(seeds), episode_length=800, batch_size=128,
                    num_minibatches=4, unroll_length=20, num_updates_per_batch=4,
                    num_resets_per_eval=0, learning_rate=job.get("learning_rate", .0003),
                    entropy_cost=.001, discounting=.99, normalize_observations=False,
                    reward_scaling=1, network_factory=network_factory,
                    wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
                    seed=seed, num_evals=5, run_evals=False, restore_params=params,
                    restore_value_fn=True, policy_params_fn=callback, progress_fn=progress)
    else:
        callback(0, None, params)
    best = max(rows, key=lambda row: row["success30"])
    summary = dict(first=rows[0]["success30"], last=rows[-1]["success30"], max=best["success30"],
                   best_step=best["step"], final_step=rows[-1]["step"],
                   elapsed_sec=time.monotonic()-started, bc_elapsed_sec=meta["bc_elapsed_sec"],
                   status="completed", job_sha256=meta["job_sha256"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
