"""Diagnostic only: keep learned residual corrections in one grasp phase."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import run_s09_learned_base_residual as learned
from brax.training.networks import FeedForwardNetwork
import jax.numpy as jp

ARCHITECTURE = "frozen learned base plus phase-gated residual diagnostic"


def gate_logits(logits,observations,phase):
    if phase not in ("before_release","after_release"):
        raise ValueError("Use one explicit correction phase")
    # Raw feature 112 is the original timestep / 800; no privileged counter.
    before = observations["state"][...,112] < 60/800
    enabled = before if phase == "before_release" else ~before
    mean = jp.where(enabled[...,None],logits[...,:17],0.)
    return jp.concatenate([mean,logits[...,17:]],axis=-1)


@contextmanager
def phase_policy(phase):
    old_init,old_architecture = learned.initialize_residual,learned.ARCHITECTURE

    def initialize(*args,**kwargs):
        net,params = old_init(*args,**kwargs)
        actor = net.policy_network
        def apply(normalizer,parameters,observations):
            return gate_logits(actor.apply(normalizer,parameters,observations),observations,phase)
        return net.replace(policy_network=FeedForwardNetwork(init=actor.init,apply=apply)),params

    learned.initialize_residual,learned.ARCHITECTURE = initialize,ARCHITECTURE
    try:
        yield
    finally:
        learned.initialize_residual,learned.ARCHITECTURE = old_init,old_architecture


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--correction-phase",choices=("before_release","after_release"),required=True)
    args,rest = p.parse_known_args()
    if "--steps" not in rest or int(rest[rest.index("--steps")+1]) != 0 or "--checkpoint" not in rest:
        p.error("Deterministic zero-update diagnostic only; not a PPO training entry")
    out = Path(rest[rest.index("--output")+1])
    sys.argv = [sys.argv[0],*rest]
    try:
        with phase_policy(args.correction_phase):
            learned.main()
    finally:
        if out.exists():
            (out/"phase_probe.json").write_text(json.dumps(dict(phase=args.correction_phase,
                gate_feature=112,boundary_step=60,optimizer_updates=0,
                limitation="Post-hoc deterministic intervention on development states, not a newly trained policy"),indent=2)+"\n")


if __name__ == "__main__":
    main()
