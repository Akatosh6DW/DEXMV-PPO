"""Reward-only ablation: delay contact-coverage saturation to a modest load.

This changes the shaping reward, not the .02 N success threshold, contact
classification, dynamics, or reference. No extra reward is paid above target.
"""

import argparse
import functools
import json
from pathlib import Path
import sys

import jax.numpy as jp

import run_s09_transition_ppo as transition
from orca_s09_env import OrcaS09Grasp
from run_s09_bc_conditioned import ModeConditionedGrasp
from retarget_s09_reference import digest


def load_coverage(loads, target):
    normalized=jp.clip(loads/target,0,1)
    return normalized[...,0]*jp.max(normalized[...,1:5],axis=-1)


class MarginRewardMixin:
    def __init__(self,*args,contact_target=.2,**kwargs):
        super().__init__(*args,**kwargs)
        if not .02 < contact_target <= 1:
            raise ValueError("Shaping load target must lie in (.02,1] N")
        self.contact_target=float(contact_target)
        with self._config.unlocked():
            self._config.reward_version="S09-R3_contact_margin"
            self._config.contact_shaping_target_n=self.contact_target

    def reset(self,rng):
        state=super().reset(rng)
        return state.replace(metrics=dict(state.metrics,coverage_margin=jp.float32(0),
                                          coverage_reward_delta=jp.float32(0)))

    def step(self,state,action):
        nxt=super().step(state,action)
        margin=load_coverage(nxt.data.audit[:5],self.contact_target)
        delta=self.coverage_weight*self.dt*(margin-nxt.metrics["coverage"])
        return nxt.replace(reward=nxt.reward+delta,metrics=dict(nxt.metrics,
            coverage_margin=margin,coverage_reward_delta=delta))


class MarginGrasp(MarginRewardMixin,OrcaS09Grasp):
    pass


class MarginConditionedGrasp(MarginRewardMixin,ModeConditionedGrasp):
    pass


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--contact-target",type=float,default=.2)
    args,rest=parser.parse_known_args()
    if not .02 < args.contact_target <= 1:
        parser.error("Shaping load target must lie in (.02,1] N")
    improvement=transition.constrained.improvement
    original_conditioned,original_env=improvement.ModeConditionedGrasp,improvement.runner.OrcaS09Grasp
    improvement.ModeConditionedGrasp=functools.partial(MarginConditionedGrasp,contact_target=args.contact_target)
    improvement.runner.OrcaS09Grasp=functools.partial(MarginGrasp,contact_target=args.contact_target)
    output=Path(rest[rest.index("--output")+1])
    original_train=improvement.runner.train.train

    def train(*values,**kwargs):
        # The lower wrappers describe their own changes; record the actual
        # supplied reward as well, before any training takes place.
        for name in ("constraint_config.json","transition_config.json"):
            path=output/name
            meta=json.loads(path.read_text())
            meta["environment_reward_version"]="S09-R3_contact_margin"
            if "reward_version" in meta:
                meta["reward_version"]="S09-R3_contact_margin"
            path.write_text(json.dumps(meta,indent=2)+"\n")
        return original_train(*values,**kwargs)

    improvement.runner.train.train=train
    sys.argv=[sys.argv[0],*rest]
    try:
        transition.main()
    finally:
        improvement.ModeConditionedGrasp=original_conditioned
        improvement.runner.OrcaS09Grasp=original_env
        improvement.runner.train.train=original_train
        path=output/"improvement_config.json"
        if path.exists():
            meta=json.loads(path.read_text())
            meta.update(reward_version="S09-R3_contact_margin",contact_shaping_target_n=args.contact_target,
                reward_formula="R0 + coverage_weight * dt * (coverage_at_target - coverage_at_0.02N)",
                success_load_threshold_n=.02,reward_source_sha256=digest(output/"sources"/Path(__file__).name))
            path.write_text(json.dumps(meta,indent=2)+"\n")


if __name__ == "__main__":
    main()
