"""Reward-only probe: discourage object motion during valid released grasp."""

import argparse
import functools
import json
import math
from pathlib import Path
import sys

import jax.numpy as jp

import run_s09_constrained_ppo as constrained
from orca_s09_env import OrcaS09Grasp
from run_s09_bc_conditioned import ModeConditionedGrasp
from retarget_s09_reference import digest


def motion_cost(linear, angular, valid, weight):
    linear_cost = jp.minimum(jp.sum(linear**2, axis=-1) / .005**2, 1.)
    angular_cost = jp.minimum(jp.sum(angular**2, axis=-1) / .1**2, 1.)
    return -.05 * weight * valid * .5 * (linear_cost + angular_cost)


class MotionRewardMixin:
    def __init__(self, *args, motion_weight=5., **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(motion_weight) or not 0 < motion_weight <= 10:
            raise ValueError("Motion weight must be finite and in (0,10]")
        self.motion_weight = float(motion_weight)
        with self._config.unlocked():
            self._config.reward_version = "S09-R5_valid_motion"
            self._config.motion_weight = self.motion_weight
            self._config.linear_motion_scale_mps = .005
            self._config.angular_motion_scale_radps = .1

    def reset(self, rng):
        state = super().reset(rng)
        return state.replace(metrics=dict(state.metrics, valid_motion_cost=jp.float32(0)))

    def step(self, state, action):
        nxt = super().step(state, action)
        velocity = nxt.data.qvel[self.dof:self.dof+6]
        valid = (nxt.metrics["released"] > 0) & (nxt.metrics["stable"] > 0)
        delta = motion_cost(velocity[:3], velocity[3:], valid, self.motion_weight)
        return nxt.replace(reward=nxt.reward+delta,
                           metrics=dict(nxt.metrics, valid_motion_cost=delta))


class MotionGrasp(MotionRewardMixin, OrcaS09Grasp):
    pass


class MotionConditionedGrasp(MotionRewardMixin, ModeConditionedGrasp):
    pass


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--motion-weight", type=float, default=5.)
    args, rest = parser.parse_known_args()
    if not math.isfinite(args.motion_weight) or not 0 < args.motion_weight <= 10:
        parser.error("Motion weight must be finite and in (0,10]")
    improvement = constrained.improvement
    old_conditioned, old_env = improvement.ModeConditionedGrasp, improvement.runner.OrcaS09Grasp
    improvement.ModeConditionedGrasp = functools.partial(MotionConditionedGrasp, motion_weight=args.motion_weight)
    improvement.runner.OrcaS09Grasp = functools.partial(MotionGrasp, motion_weight=args.motion_weight)
    output = Path(rest[rest.index("--output")+1])
    old_train = improvement.runner.train.train

    def train(*values, **kwargs):
        path = output / "constraint_config.json"
        metadata = json.loads(path.read_text())
        metadata["reward_version"] = "S09-R5_valid_motion"
        path.write_text(json.dumps(metadata, indent=2) + "\n")
        return old_train(*values, **kwargs)

    improvement.runner.train.train = train
    sys.argv = [sys.argv[0], *rest]
    try:
        constrained.main()
    finally:
        improvement.ModeConditionedGrasp = old_conditioned
        improvement.runner.OrcaS09Grasp = old_env
        improvement.runner.train.train = old_train
        path = output / "improvement_config.json"
        if path.exists():
            metadata = json.loads(path.read_text())
            metadata.update(reward_version="S09-R5_valid_motion", motion_weight=args.motion_weight,
                motion_formula="R0 - .05*w*valid*.5*(min(||v||^2/.005^2,1)+min(||omega||^2/.1^2,1))",
                success_criterion_unchanged=True,
                reward_source_sha256=digest(output / "sources" / Path(__file__).name))
            path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
