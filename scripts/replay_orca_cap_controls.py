#!/usr/bin/env python3
"""Replay labeled control references in the cap environment; no PPO training."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from orca_cap_env import OrcaCap, default_config
from train_orca_cap import summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--kind", choices=("procedural_ik", "video_grip_transfer"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--noise", type=float, default=.002)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = default_config()
    config.initial_noise = args.noise
    env = OrcaCap(args.model, config)
    with np.load(args.reference) as z:
        controls = jp.array(z["ctrl"], jp.float32)
    controls = controls[jp.minimum(jp.arange(config.episode_length), len(controls)-1)]
    actions = jax.vmap(env.ctrl_to_action)(controls)

    def episode(key):
        state = env.reset(key)

        def step(carry, action):
            state, alive = carry
            nxt = env.step(state, action)
            alive &= nxt.done == 0
            return (nxt, alive), dict(obs=state.obs["state"], action=action, qpos=nxt.data.qpos,
                                      qvel=nxt.data.qvel, ctrl=nxt.data.ctrl, alive=alive,
                                      reward=jp.where(alive, nxt.reward, 0), **nxt.metrics)
        return jax.lax.scan(step, (state, jp.bool_(True)), actions)[1]

    keys = jax.vmap(jax.random.PRNGKey)(jp.arange(args.count))
    trace = jax.device_get(jax.jit(jax.vmap(episode))(keys))
    summary, holds = summarize(trace, env)
    np.savez_compressed(args.output / "trajectories.npz", **trace)
    report = dict(kind=args.kind, is_ppo=False, is_cap_twist_video=False,
                  source=str(args.reference.resolve()),
                  source_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  model=str(args.model.resolve()), model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  count=args.count, initial_noise=args.noise, summary=summary, holds=holds.tolist())
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
