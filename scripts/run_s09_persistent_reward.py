"""Reward-only ablation: prefer uninterrupted, physically qualified holding.

This is an additive training signal, NOT a change to the 30-second success
criterion, reference bank, action composition, dynamics, or observations.
"""

import argparse
import functools
import json
from pathlib import Path
import sys

import jax.numpy as jp

import run_s09_ppo_improvement as improvement
from orca_s09_env import OrcaS09Grasp
from run_s09_bc_conditioned import ModeConditionedGrasp
from retarget_s09_reference import digest


def hold_bonus(hold_steps, valid, clean_history, weight, dt=.05):
    return weight * dt * jp.minimum(hold_steps/600., 1.) * valid * clean_history


class PersistentRewardMixin:
    def __init__(self, *args, hold_weight=15., **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < hold_weight <= 100:
            raise ValueError("Use a finite positive hold weight at most 100")
        self.hold_weight = float(hold_weight)
        with self._config.unlocked():
            self._config.reward_version = "S09-R2_qualified_hold_progress"
            self._config.hold_progress_weight = self.hold_weight

    def reset(self, rng):
        state = super().reset(rng)
        return state.replace(metrics=dict(state.metrics, hold_progress_bonus=jp.float32(0)))

    def step(self, state, action):
        nxt = super().step(state, action)
        clean = ((nxt.info["initial_penetration"] <= .001) &
                 (nxt.info["max_depth"] <= .003) & ~nxt.info["ever_forbidden"])
        bonus = hold_bonus(nxt.info["hold_steps"], nxt.metrics["stable"], clean, self.hold_weight, self.dt)
        return nxt.replace(reward=nxt.reward+bonus,
            metrics=dict(nxt.metrics, hold_progress_bonus=bonus))


class PersistentGrasp(PersistentRewardMixin, OrcaS09Grasp):
    pass


class PersistentConditionedGrasp(PersistentRewardMixin, ModeConditionedGrasp):
    pass


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--hold-bonus-weight", type=float, default=15.)
    a, rest = p.parse_known_args()
    if not 0 < a.hold_bonus_weight <= 100:
        p.error("Invalid persistence reward weight")
    output = Path(rest[rest.index("--output")+1])
    improvement.ModeConditionedGrasp = functools.partial(PersistentConditionedGrasp, hold_weight=a.hold_bonus_weight)
    improvement.runner.OrcaS09Grasp = functools.partial(PersistentGrasp, hold_weight=a.hold_bonus_weight)
    sys.argv = [sys.argv[0], *rest]
    try:
        improvement.main()
    finally:
        meta_path = output/"improvement_config.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta.update(reward_version="S09-R2_qualified_hold_progress", hold_progress_weight=a.hold_bonus_weight,
                reward_formula="R0 + weight * dt * min(continuous_valid_steps/600,1) * valid * clean_history")
            snapshot = output/"sources"/Path(__file__).name
            meta["reward_source_sha256"] = digest(snapshot if snapshot.exists() else Path(__file__))
            meta_path.write_text(json.dumps(meta,indent=2)+"\n")


if __name__ == "__main__":
    main()
