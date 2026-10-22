"""Training-only velocity disturbances for full-target BC+PPO.

The original evaluation environment, reward, actor and physics model remain
unchanged. Random disturbances are not a learned adversary or RARL replication.
"""

import argparse
import json
from pathlib import Path
import sys

import run_s09_constrained_ppo as constrained
from mujoco_playground._src import wrapper
from orca_native_physics import STATE_SPEC
from retarget_s09_reference import digest
import jax
import jax.numpy as jp
import mujoco
import numpy as np


class DisturbanceTraining(wrapper.Wrapper):
    def __init__(self, env, probability=.5, speed=.02):
        super().__init__(env)
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Probability must lie in [0,1]")
        if not np.isfinite(speed) or not 0 < speed <= .02:
            raise ValueError("Velocity increment must lie in (0,.02] m/s")
        self.probability, self.speed = float(probability), float(speed)
        self.native = env.native
        self.dof = env.dof
        self.spec = (jax.ShapeDtypeStruct((2*self.native.state_size,), np.uint32),
                     jax.ShapeDtypeStruct((self.native.model.nv,), np.float32))

    def reset(self, rng):
        state = self.env.reset(rng)
        return state.replace(metrics=dict(state.metrics, training_kick=jp.float32(0)))

    def _single(self, words, velocity, delta):
        if not np.any(delta):
            return words.copy(), velocity.copy()
        with self.native.lock:
            data = self.native.restore(words)
            data.qvel[self.dof:self.dof+3] += delta
            packed = np.empty(self.native.state_size, np.float64)
            mujoco.mj_getState(self.native.model, data, packed, STATE_SPEC)
            return packed.view(np.uint32).copy(), data.qvel.astype(np.float32)

    def _batch(self, words, velocity, delta):
        words, velocity, delta = map(np.asarray, (words, velocity, delta))
        shape = words.shape[:-1]
        if not shape:
            return self._single(words, velocity, delta)
        values = [self._single(words[i], velocity[i], delta[i]) for i in np.ndindex(shape)]
        return tuple(np.stack([x[k] for x in values]).reshape((*shape, *values[0][k].shape))
                     for k in (0,1))

    def step(self, state, action):
        nxt = self.env.step(state, action)
        if self.probability == 0:
            return nxt.replace(metrics=dict(nxt.metrics, training_kick=jp.float32(0)))
        coin, direction = jax.random.split(jax.random.fold_in(nxt.info["rng"], 264001))
        selected = ((nxt.info["step"] == 40) & (nxt.done == 0)
                    & (jax.random.uniform(coin) < self.probability))
        directions = jp.asarray([[1.,0,0],[-1.,0,0],[0,0,1.],[0,0,-1.]])
        delta = jp.where(selected, self.speed*directions[jax.random.randint(direction, (), 0, 4)], 0.)
        words, velocity = jax.pure_callback(self._batch, self.spec,
            nxt.data.integration_words, nxt.data.qvel, delta, vmap_method="broadcast_all")
        data = nxt.data.replace(integration_words=words, qvel=velocity)
        # Preserve time/pose and delayed contact features exactly. The kick
        # happens after this step's reward, before the next policy observation.
        actor = nxt.obs["state"].at[71:74].set(jp.where(selected,
            velocity[self.dof:self.dof+3]*2, nxt.obs["state"][71:74]))
        privileged = nxt.obs["privileged_state"].at[:123].set(actor)
        return nxt.replace(data=data, obs=dict(state=actor, privileged_state=privileged),
                           metrics=dict(nxt.metrics, training_kick=selected.astype(jp.float32)))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--disturbance-probability", type=float, default=.5)
    parser.add_argument("--disturbance-speed", type=float, default=.02)
    args, rest = parser.parse_known_args()
    if rest[rest.index("--method")+1] != "guided" or "--checkpoint" not in rest:
        parser.error("Use a full-target BC checkpoint; no new BC data in this controlled PPO probe")
    if not 0 <= args.disturbance_probability <= 1 or not 0 < args.disturbance_speed <= .02:
        parser.error("Invalid probability or velocity bound")
    runner = constrained.improvement.runner
    output = Path(rest[rest.index("--output")+1])
    original = runner.train.train

    def train(*values, **kwargs):
        kwargs["environment"] = DisturbanceTraining(kwargs["environment"],
            args.disturbance_probability, args.disturbance_speed)
        metadata = dict(probability=args.disturbance_probability, speed_mps=args.disturbance_speed,
            phase_step=40, directions=["+x","-x","+z","-z"], rng_fold_in=264001,
            training_only=True, evaluation_wrapped=False, reward="original S09-R0 unchanged",
            action_architecture="single learned full target, not residual",
            source_sha256=digest(output/"sources"/Path(__file__).name),
            limitation="Random training disturbance, not RARL or a new video-derived teacher")
        (output/"disturbance_training_config.json").write_text(json.dumps(metadata, indent=2)+"\n")
        return original(*values, **kwargs)

    runner.train.train = train
    sys.argv = [sys.argv[0], *rest]
    try:
        constrained.main()
    finally:
        runner.train.train = original


if __name__ == "__main__":
    main()
