"""Reward-only test: penalize breaking an existing released valid-contact streak.

R0 pays for each valid instant; strict success requires an uninterrupted streak.
The added event penalty does not change physics, observations, or success rules.
"""

import argparse
import functools
import json
from pathlib import Path
import sys

import jax.numpy as jp

import run_s09_constrained_ppo as constrained
from orca_s09_env import OrcaS09Grasp
from run_s09_bc_conditioned import ModeConditionedGrasp
from retarget_s09_reference import digest


def break_cost(previous_hold,released,valid,coefficient):
    broken=(previous_hold>0)&released&~valid
    return -coefficient*broken.astype(jp.float32)


class BreakRewardMixin:
    def __init__(self,*args,break_penalty=5.,**kwargs):
        super().__init__(*args,**kwargs)
        if not 0<break_penalty<=50:
            raise ValueError("Event penalty must lie in (0,50]")
        self.break_penalty=float(break_penalty)
        with self._config.unlocked():
            self._config.reward_version="S09-R4_contact_break"
            self._config.contact_break_penalty=self.break_penalty

    def reset(self,rng):
        state=super().reset(rng)
        return state.replace(metrics=dict(state.metrics,contact_break_cost=jp.float32(0)))

    def step(self,state,action):
        nxt=super().step(state,action)
        delta=break_cost(state.info["hold_steps"],nxt.metrics["released"]>0,
                         nxt.metrics["stable"]>0,self.break_penalty)
        return nxt.replace(reward=nxt.reward+delta,
                           metrics=dict(nxt.metrics,contact_break_cost=delta))


class BreakGrasp(BreakRewardMixin,OrcaS09Grasp):
    pass


class BreakConditionedGrasp(BreakRewardMixin,ModeConditionedGrasp):
    pass


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--break-penalty",type=float,default=5.)
    args,rest=parser.parse_known_args()
    if not 0<args.break_penalty<=50:
        parser.error("Event penalty must lie in (0,50]")
    improvement=constrained.improvement
    old_conditioned,old_env=improvement.ModeConditionedGrasp,improvement.runner.OrcaS09Grasp
    improvement.ModeConditionedGrasp=functools.partial(BreakConditionedGrasp,break_penalty=args.break_penalty)
    improvement.runner.OrcaS09Grasp=functools.partial(BreakGrasp,break_penalty=args.break_penalty)
    output=Path(rest[rest.index("--output")+1])
    old_train=improvement.runner.train.train

    def train(*values,**kwargs):
        path=output/"constraint_config.json"
        metadata=json.loads(path.read_text())
        metadata["reward_version"]="S09-R4_contact_break"
        path.write_text(json.dumps(metadata,indent=2)+"\n")
        return old_train(*values,**kwargs)

    improvement.runner.train.train=train
    sys.argv=[sys.argv[0],*rest]
    try:
        constrained.main()
    finally:
        improvement.ModeConditionedGrasp=old_conditioned
        improvement.runner.OrcaS09Grasp=old_env
        improvement.runner.train.train=old_train
        path=output/"improvement_config.json"
        if path.exists():
            metadata=json.loads(path.read_text())
            metadata.update(reward_version="S09-R4_contact_break",contact_break_penalty=args.break_penalty,
                reward_formula="R0 - penalty * (previous_hold_steps > 0 and released and not valid)",
                penalty_is_per_event_not_per_second=True,success_criterion_unchanged=True,
                reward_source_sha256=digest(output/"sources"/Path(__file__).name))
            path.write_text(json.dumps(metadata,indent=2)+"\n")


if __name__ == "__main__":
    main()
