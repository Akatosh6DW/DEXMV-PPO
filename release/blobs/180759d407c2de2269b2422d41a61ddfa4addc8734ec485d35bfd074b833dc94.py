"""Versioned reward ablation and stochastic rollout diagnostics for Orca."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import time

from brax.training.agents.ppo import networks
import jax
import jax.numpy as jp
import numpy as np

from orca_residual_env import OrcaResidualGrasp
from orca_residual_metrics import summarize
from train_orca_residual import initialize_zero_mean

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/orca30/dexmv_slow_acquisition_o21/reference.npz"
DATASET = ROOT / "artifacts/orca_residual/shared_cube_demos/trajectories.npz"


def task_reward(stable, done, dt):
    # Remove geometric/contact shaping, retain exactly the old valid-hold signal.
    return (15 * stable - 50 * done) * dt


class RewardAblationGrasp(OrcaResidualGrasp):
    def __init__(self, arm, reward_mode="shaped"):
        if reward_mode not in ("shaped", "hold_only"):
            raise ValueError("Unknown reward mode")
        self.reward_mode = reward_mode
        super().__init__(ROOT / "artifacts/orca_residual/models/cube.xml",
                         "pure" if arm == "bc" else arm, REFERENCE, .08)

    def step(self, state, action):
        nxt = super().step(state, action)
        if self.reward_mode == "hold_only":
            nxt = nxt.replace(reward=task_reward(nxt.metrics["stable"], nxt.done, self.dt))
        return nxt


def make_evaluator(env, net, seeds, stochastic=False):
    keys = jax.vmap(jax.random.PRNGKey)(jp.asarray(seeds, jp.int32))
    inference = networks.make_inference_fn(net)

    def evaluate(params):
        policy = inference(params, deterministic=not stochastic)

        def episode(key):
            state = env.reset(key)
            action_key = jax.random.fold_in(key, 987654)

            def step(carry, _):
                state, alive, key = carry
                key, subkey = jax.random.split(key)
                action = policy(state.obs, subkey)[0]
                nxt = env.step(state, action)
                next_alive = alive & (nxt.done == 0)
                trace = dict(obs=state.obs["state"], action=action, alive=next_alive,
                             active=alive, hold=jp.where(next_alive, nxt.metrics["hold_sec"], 0),
                             reward=jp.where(alive, nxt.reward, 0),
                             qpos=nxt.data.qpos, qvel=nxt.data.qvel, ctrl=nxt.data.ctrl,
                             mocap=nxt.data.mocap_pos, forbidden=nxt.metrics["forbidden_contact"],
                             contacts=nxt.metrics["contact_count"], thumb=nxt.metrics["thumb_contact"],
                             released=nxt.metrics["released"], cube_z=nxt.metrics["cube_z"],
                             spawn_z=nxt.info["cube_spawn_pos"][2])
                for name in ("video_base", "requested_residual", "effective_residual", "desired_ctrl"):
                    trace[name] = nxt.info[name]
                return (nxt, next_alive, key), trace

            return jax.lax.scan(step, (state, jp.bool_(True), action_key), None, length=800)[1]
        return jax.vmap(episode)(keys)
    return jax.jit(evaluate)


def score(trace, env):
    row, holds, success = summarize(trace, env.arm, env.residual_bound, env._lowers, env._uppers)
    active = trace["active"]
    correction = trace["requested_residual"][active]
    row.update(final_hold30=float(np.mean(trace["hold"][:, -1] >= 30)),
               active_residual_rms_rad=float(np.sqrt(np.mean(correction ** 2))),
               active_residual_saturation=float(np.mean(np.abs(correction) >= .99 * env.residual_bound)),
               active_action_rms=float(np.sqrt(np.mean(trace["action"][active] ** 2))))
    return row, holds, success


def snapshot(output, config, env):
    sources = output / "sources"
    sources.mkdir()
    names = ["orca_guidance_recheck.py", "train_orca_guidance_recheck.py",
             "train_orca_residual.py", "orca_residual_env.py", "orca_residual_controls.py",
             "orca_residual_metrics.py", "orca_benchmark_env.py", "train_orca_comparison.py"]
    config["source_sha256"] = {}
    for name in names:
        source = ROOT / "scripts" / name
        if source.exists():
            data = source.read_bytes()
            (sources / name).write_bytes(data)
            config["source_sha256"][name] = hashlib.sha256(data).hexdigest()
    for name, source in (("reference", REFERENCE), ("dataset", DATASET),
                         ("model", ROOT / "artifacts/orca_residual/models/cube.xml")):
        config[name + "_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    config.update(jax_version=jax.__version__, backend=jax.default_backend(),
                  precision=jax.config.jax_default_matmul_precision)
    (output / "run_config.json").write_text(json.dumps(config, indent=2))
    (output / "env_config.json").write_text(env._config.to_json(indent=2))


def diagnostic_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--eval-start", type=int, default=4000)
    parser.add_argument("--eval-count", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "artifacts/orca30/jax_cache"))
    env = RewardAblationGrasp("residual")
    net, _ = initialize_zero_mean(91, .7)
    seeds = list(range(args.eval_start, args.eval_start + args.eval_count))
    evaluate = {mode: make_evaluator(env, net, seeds, mode == "stochastic")
                for mode in ("deterministic", "stochastic")}
    snapshot(args.output, dict(experiment="untrained_residual_noise", seeds=seeds,
                              stds=[.7, .1, .03], seed=91), env)
    rows = []
    for std in (.7, .1, .03):
        _, params = initialize_zero_mean(91, std)
        for mode in evaluate:
            started = time.monotonic()
            trace = jax.device_get(evaluate[mode](params))
            row, holds, success = score(trace, env)
            row.update(std=std, mode=mode, elapsed_sec=time.monotonic() - started)
            rows.append(row)
            target = args.output / f"std{std}_{mode}"
            target.mkdir()
            np.savez_compressed(target / "trajectories.npz", **trace)
            (target / "episodes.json").write_text(json.dumps(dict(
                seeds=seeds, hold_sec=holds.tolist(), success30=success.tolist()), indent=2))
            with (args.output / "metrics.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    diagnostic_main()
