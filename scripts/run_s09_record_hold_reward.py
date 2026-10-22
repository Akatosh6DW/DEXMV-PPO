"""Reward-only probe: pay for extending the episode's best valid hold streak.

The undiscounted main reward telescopes to 15 * dt * best_hold_steps.
This is still a surrogate, not the full terminal success criterion: shaping,
discounting, and the original cumulative contact-quality gates still matter.
"""

import argparse
import json
from pathlib import Path
import sys

import jax.numpy as jp

import run_s09_constrained_ppo as constrained
from run_s09_bc_conditioned import ModeConditionedGrasp
from retarget_s09_reference import digest


def record_hold_delta(previous_best, current_best, valid, dt=.05):
    increment = current_best - previous_best
    return 15 * dt * (increment - jp.asarray(valid, jp.float32))


class RecordHoldGrasp(ModeConditionedGrasp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with self._config.unlocked():
            self._config.reward_version = "S09-R6_record_hold"

    def reset(self, rng):
        state = super().reset(rng)
        return state.replace(metrics=dict(state.metrics, record_hold_delta=jp.float32(0)))

    def step(self, state, action):
        nxt = super().step(state, action)
        delta = record_hold_delta(state.info["best_steps"], nxt.info["best_steps"],
                                  nxt.metrics["stable"], self.dt)
        return nxt.replace(reward=nxt.reward + delta,
            metrics=dict(nxt.metrics, record_hold_delta=delta))


def record_metadata(output):
    source = output / "sources" / Path(__file__).name
    for name in ("constraint_config.json", "improvement_config.json"):
        path = output / name
        if path.exists():
            metadata = json.loads(path.read_text())
            metadata.update(reward_version="S09-R6_record_hold",
                reward_formula="R0 + 15 * dt * (next_best_steps - previous_best_steps - valid)",
                success_criterion_unchanged=True, physics_and_observations_unchanged=True,
                reward_source_sha256=digest(source if source.exists() else Path(__file__)))
            path.write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    _, rest = parser.parse_known_args()
    if "--method" not in rest or rest[rest.index("--method") + 1] != "guided":
        parser.error("Use the original single full-target BC+PPO actor")
    if any(option in rest for option in ("--teacher-kl", "--ppo-discount", "--ppo-initial-std")):
        parser.error("Do not mix this reward probe with teacher, discount, or exploration changes")
    output = Path(rest[rest.index("--output") + 1])
    improvement = constrained.improvement
    runner = improvement.runner
    old_train = runner.train.train
    old_conditioned, old_env = improvement.ModeConditionedGrasp, runner.OrcaS09Grasp

    def train(*values, **kwargs):
        record_metadata(output)
        return old_train(*values, **kwargs)

    runner.train.train = train
    improvement.ModeConditionedGrasp = RecordHoldGrasp
    sys.argv = [sys.argv[0], *rest]
    try:
        constrained.main()
    finally:
        runner.train.train = old_train
        improvement.ModeConditionedGrasp, runner.OrcaS09Grasp = old_conditioned, old_env
        record_metadata(output)


if __name__ == "__main__":
    main()
