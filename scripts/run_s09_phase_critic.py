"""Isolate acquisition/holding value functions without changing residual actors."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

from brax.training.networks import FeedForwardNetwork
import jax
import jax.numpy as jp
import numpy as np

import run_s09_learned_residual_experts as experts
from retarget_s09_reference import digest

ARCHITECTURE = experts.ARCHITECTURE + " and two release-phase value critics"


def holding_phase(observations):
    return (observations["privileged_state"][...,93] >= 1).astype(jp.int32)


@contextmanager
def phase_critics(enabled):
    if not enabled:
        yield
        return
    previous = experts.residual_expert_factory,experts.SINGLE_INITIALIZE,experts.ARCHITECTURE

    def factory(base):
        make = previous[0](base)

        def network(*args,**kwargs):
            net = make(*args,**kwargs)
            value = net.value_network

            def init(key):
                params = value.init(key)
                return {"phases":tuple(jax.tree_util.tree_map(jp.array,params) for _ in range(2))}

            def apply(stats,params,observations):
                if len(params["phases"]) != 2:
                    raise ValueError("Acquisition and holding critics required")
                values = jp.stack([value.apply(stats,p,observations) for p in params["phases"]],axis=-1)
                return jp.take_along_axis(values,holding_phase(observations)[...,None],axis=-1)[...,0]

            return net.replace(value_network=FeedForwardNetwork(init=init,apply=apply))
        return network

    def initialize(*args,**kwargs):
        net,params = previous[1](*args,**kwargs)
        value = {"phases":tuple(jax.tree_util.tree_map(jp.array,params[2]) for _ in range(2))}
        return net,(params[0],params[1],value)

    experts.residual_expert_factory,experts.SINGLE_INITIALIZE,experts.ARCHITECTURE = factory,initialize,ARCHITECTURE
    try:
        yield
    finally:
        experts.residual_expert_factory,experts.SINGLE_INITIALIZE,experts.ARCHITECTURE = previous


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--critic-routing",choices=("shared","phase"),required=True)
    args,rest = p.parse_known_args()
    if "--checkpoint" in rest:
        p.error("Start this matched probe from the frozen base")
    enabled = args.critic_routing == "phase"
    out = Path(rest[rest.index("--output")+1])
    runner = experts.learned.runner
    original,argv = runner.train.train,sys.argv
    record = dict(routing=args.critic_routing,critic_count=2 if enabled else 1,
        phase_rule="raw privileged_state[93] >= 1; original time progress reaches 3 seconds",
        release_sec=3,source_sha256=digest(Path(__file__)),
        limitation="Twofold critic capacity plus temporal routing; not capacity matched",
        bootstrap="Original Brax GAE uses the selected next-state critic, including across the release boundary",
        unchanged=["six residual actors","frozen base","R0","physical release","reward","action composition"])

    def train(*values,**kwargs):
        params = kwargs["restore_params"][2]
        critics = params["phases"] if enabled else (params,)
        record["parameters_per_critic"] = [sum(x.size for x in jax.tree_util.tree_leaves(p)) for p in critics]
        record["initial_critics_identical"] = all(all(np.array_equal(x,y) for x,y in
            zip(jax.tree_util.tree_leaves(critics[0]),jax.tree_util.tree_leaves(p))) for p in critics)
        assert record["initial_critics_identical"]
        (out/"phase_critic_config.json").write_text(json.dumps(record,indent=2)+"\n")
        return original(*values,**kwargs)

    runner.train.train,sys.argv = train,[sys.argv[0],*rest]
    try:
        with phase_critics(enabled):
            experts.main()
    finally:
        runner.train.train,sys.argv = original,argv
        if out.exists():
            (out/"phase_critic_config.json").write_text(json.dumps(record,indent=2)+"\n")


if __name__=="__main__":
    main()
