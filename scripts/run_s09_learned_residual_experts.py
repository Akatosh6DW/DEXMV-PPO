"""Separate six residual actors while retaining the same frozen learned base."""

from contextlib import contextmanager
import json
from pathlib import Path
import sys

import run_s09_learned_base_residual as learned
from run_s09_mode_experts import expand_parameters
from brax.training.networks import FeedForwardNetwork
import jax
import jax.numpy as jp

ARCHITECTURE = "frozen learned full-target experts plus six pose-routed residual PPO actors"
SINGLE_FACTORY = learned.residual_factory
SINGLE_INITIALIZE = learned.initialize_residual


def residual_expert_factory(base_action):
    single = SINGLE_FACTORY(base_action)

    def factory(observation_size,action_size,**kwargs):
        net = single(observation_size,action_size,**kwargs)
        actor = net.policy_network

        def init(key):
            params = actor.init(key)
            return {"experts":tuple(jax.tree_util.tree_map(jp.array,params) for _ in range(6))}

        def apply(normalizer,params,observations):
            if len(params["experts"]) != 6:
                raise ValueError("Six residual experts required")
            logits = jp.stack([actor.apply(normalizer,p,observations) for p in params["experts"]],axis=-2)
            mode = jp.argmax(observations["state"][...,95:101],axis=-1)
            return jp.take_along_axis(logits,mode[...,None,None],axis=-2)[...,0,:]

        return net.replace(policy_network=FeedForwardNetwork(init=init,apply=apply))
    return factory


@contextmanager
def residual_experts():
    previous = learned.residual_factory,learned.initialize_residual,learned.ARCHITECTURE
    base = {}

    def factory(base_action):
        base["action"] = base_action
        return residual_expert_factory(base_action)

    def initialize(factory,params,seed,std):
        _,single_params = SINGLE_INITIALIZE(SINGLE_FACTORY(base["action"]),params,seed,std)
        return factory(dict(state=(123,),privileged_state=(140,)),17),expand_parameters(single_params)

    learned.residual_factory,learned.initialize_residual,learned.ARCHITECTURE = factory,initialize,ARCHITECTURE
    try:
        yield
    finally:
        learned.residual_factory,learned.initialize_residual,learned.ARCHITECTURE = previous


def main():
    out = Path(sys.argv[sys.argv.index("--output")+1])
    old_train = learned.runner.train.train
    record = dict(residual_expert_count=6,shared_residual_parameters=False,
        routing="remembered initial pose argmax(raw_state[95:101])",
        action_law_unchanged=True,base_networks_unchanged=True,
        limitation="Sixfold residual actor capacity; not capacity-matched to a single residual actor")

    def train(*args,**kwargs):
        params = kwargs["restore_params"][1]
        record["parameters_per_residual_actor"] = [sum(x.size for x in jax.tree_util.tree_leaves(p)) for p in params["experts"]]
        (out/"residual_expert_config.json").write_text(json.dumps(record,indent=2)+"\n")
        return old_train(*args,**kwargs)

    learned.runner.train.train = train
    try:
        with residual_experts():
            learned.main()
    finally:
        learned.runner.train.train = old_train
        if out.exists():
            (out/"residual_expert_config.json").write_text(json.dumps(record,indent=2)+"\n")


if __name__ == "__main__":
    main()
