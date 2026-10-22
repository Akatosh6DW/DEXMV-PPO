"""Optional R1 reward-only continuation, gated on the completed R0 comparison."""

import argparse
import functools
import json
from pathlib import Path
import sys

import jax.numpy as jp

import run_s09_job as runner
from orca_s09_env import OrcaS09Grasp
from s09_suspension import OUT


class PenetrationPenaltyGrasp(OrcaS09Grasp):
    """Leave dynamics, action composition and success criteria unchanged."""

    def __init__(self, *args, penalty_weight=5., **kwargs):
        super().__init__(*args, **kwargs)
        if penalty_weight <= 0:
            raise ValueError("Use R0 directly for zero penalty")
        self.penalty_weight = float(penalty_weight)
        with self._config.unlocked():
            self._config.reward_version = "S09-R1_penetration"
            self._config.penetration_penalty = dict(weight=penalty_weight, start_m=.0015,
                                                   scale_m=.0015, maximum=4.)

    def reset(self, rng):
        state = super().reset(rng)
        return state.replace(metrics=dict(state.metrics, penetration_penalty=jp.float32(0)))

    def step(self, state, action):
        nxt = super().step(state, action)
        penalty = jp.clip((nxt.data.audit[6]-.0015)/.0015, 0, 4)
        return nxt.replace(reward=nxt.reward-self.penalty_weight*penalty*self.dt,
                           metrics=dict(nxt.metrics, penetration_penalty=penalty))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--penalty-weight", type=float, default=5.)
    parser.add_argument("--output", type=Path, required=True)
    args, rest = parser.parse_known_args()
    audit = json.loads((OUT/"report/matrix_verification.json").read_text())
    if audit.get("status")!="matrix_verified" or audit.get("jobs")!=50 or audit.get("episodes")!=2560:
        raise ValueError("Complete and verify R0 before the reward-iteration experiment")
    if not args.output.resolve().is_relative_to(OUT/"reward_iteration"):
        raise ValueError("R1 output must remain separate from the frozen R0 runs")
    runner.OrcaS09Grasp = functools.partial(PenetrationPenaltyGrasp, penalty_weight=args.penalty_weight)
    sys.argv = [sys.argv[0],*rest,"--output",str(args.output)]
    runner.main()


if __name__=="__main__":
    main()
