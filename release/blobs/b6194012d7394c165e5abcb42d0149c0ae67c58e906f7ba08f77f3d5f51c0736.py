"""Fresh reward-by-guidance experiment; no persistent BC loss or old edits."""

import argparse
import csv
import functools
import json
from pathlib import Path
import pickle
import sys
import time

from brax.training.agents.ppo import train
import jax
import numpy as np

from mujoco_playground._src import wrapper
from orca_guidance_recheck import DATASET, ROOT, RewardAblationGrasp, make_evaluator, score, snapshot
from train_orca_comparison import fit_bc, network_factory
from train_orca_residual import initialize_zero_mean


def curve_summary(rows):
    steps = np.asarray([r["step"] for r in rows])
    scores = np.asarray([r["success30"] for r in rows])
    auc = float(np.trapezoid(scores, steps) / steps[-1]) if steps[-1] else float(scores[0])
    hit = next((int(steps[i - 1]) for i in range(1, len(rows))
                if scores[i - 1] >= .75 and scores[i] >= .75), None)
    return dict(first=float(scores[0]), last=float(scores[-1]), max=float(scores.max()),
                best_step=int(steps[np.argmax(scores)]), auc=auc,
                gain_auc=auc - float(scores[0]), learning_gain=float(scores[-1] - scores[0]),
                sustained75_step=hit, censored=hit is None,
                starting75=bool(scores[0] >= .75))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("pure", "bc", "residual"), required=True)
    parser.add_argument("--reward", choices=("shaped", "hold_only"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=163840)
    parser.add_argument("--evals", type=int, default=3)
    parser.add_argument("--eval-start", type=int, default=4100)
    parser.add_argument("--eval-count", type=int, default=16)
    parser.add_argument("--init-std", type=float, default=.1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    if args.steps < 0 or (args.checkpoint and args.steps != 0):
        parser.error("Checkpoints are evaluation only; steps must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "artifacts/orca30/jax_cache"))
    start = time.monotonic()
    env = RewardAblationGrasp(args.arm, args.reward)
    net, params = initialize_zero_mean(args.seed, args.init_std)
    bc_sec = 0.
    if args.checkpoint:
        with args.checkpoint.open("rb") as f:
            params = pickle.load(f)
    elif args.arm == "bc":
        before = time.monotonic()
        params, losses = fit_bc(net, params, DATASET, 2000, args.seed)
        bc_sec = time.monotonic() - before
        (args.output / "bc_loss.json").write_text(json.dumps(losses, indent=2))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(command=sys.argv, bc_sec=bc_sec, demo_loss_weight=0,
                  discounting=.99, entropy_cost=.001, num_envs=128, batch_size=128,
                  num_minibatches=4, unroll_length=20, num_updates_per_batch=4,
                  bc_updates=2000 if args.arm == "bc" and not args.checkpoint else 0,
                  primary="30s continuous clean hold plus valid final frame; unchanged from R08",
                  reward_note="hold_only is dense valid-hold reward, NOT terminal-only sparse reward")
    snapshot(args.output, config, env)
    seeds = list(range(args.eval_start, args.eval_start + args.eval_count))
    evaluate = make_evaluator(env, net, seeds, args.stochastic)
    rows = []

    def callback(step, make_policy, current):
        del make_policy
        trace = jax.device_get(evaluate(current))
        for name, values in trace.items():
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite trace: {name}")
        row, holds, success = score(trace, env)
        row = dict(step=int(step), **row, elapsed_sec=time.monotonic() - start)
        rows.append(row)
        with (args.output / "metrics.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        checkpoint = args.output / f"step_{int(step):09d}"
        checkpoint.mkdir()
        with (checkpoint / "params.pkl").open("wb") as f:
            pickle.dump(jax.device_get(current), f)
        np.savez_compressed(checkpoint / "trajectories.npz", **trace)
        (checkpoint / "episodes.json").write_text(json.dumps(dict(
            seeds=seeds, hold_sec=holds.tolist(), success30=success.tolist()), indent=2))
        print(json.dumps(row), flush=True)

    def progress(step, metrics):
        values = {k: float(v) for k, v in metrics.items()}
        if not np.isfinite(list(values.values())).all():
            raise ValueError("Nonfinite PPO metrics")
        with (args.output / "ppo_metrics.jsonl").open("a") as f:
            f.write(json.dumps(dict(step=int(step), **values)) + "\n")

    if args.steps == 0:
        callback(0, None, params)
    else:
        train.train(environment=env, num_timesteps=args.steps, num_envs=128,
                    num_eval_envs=args.eval_count, episode_length=800,
                    batch_size=128, num_minibatches=4, unroll_length=20,
                    num_updates_per_batch=4, num_resets_per_eval=0,
                    learning_rate=args.learning_rate, entropy_cost=.001,
                    discounting=.99, normalize_observations=False, reward_scaling=1,
                    network_factory=network_factory,
                    wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
                    seed=args.seed, num_evals=args.evals, run_evals=False,
                    restore_params=params, restore_value_fn=True,
                    policy_params_fn=callback, progress_fn=progress)
    summary = curve_summary(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("DONE", args.output, summary, flush=True)


if __name__ == "__main__":
    main()
