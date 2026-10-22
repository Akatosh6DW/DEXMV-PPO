"""Fit the critic before allowing updates to the already trained BC actor.

Warmup transitions count toward the total online budget. Actor parameters AND
its Adam moments remain unchanged until warmup ends; inference is unchanged.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import NamedTuple

import jax
import jax.numpy as jp
import optax

import run_s09_constrained_ppo as constrained
from retarget_s09_reference import digest


class DelayedState(NamedTuple):
    count: object
    inner: object


def delayed_adam(rate, warmup_updates):
    adam = optax.adam(rate)

    def init(params):
        return DelayedState(jp.int32(0), adam.init(params))

    def update(grads, state, params=None):
        updates, inner = jax.lax.cond(state.count < warmup_updates,
            lambda _: (jax.tree_util.tree_map(jp.zeros_like, grads), state.inner),
            lambda _: adam.update(grads, state.inner, params), operand=None)
        return updates, DelayedState(state.count+1, inner)
    return optax.GradientTransformation(init, update)


def warmup_optimizer(actor_rate, critic_rate, updates):
    def labels(params):
        return params.replace(policy=jax.tree_util.tree_map(lambda _:"actor",params.policy),
                              value=jax.tree_util.tree_map(lambda _:"critic",params.value))
    return optax.multi_transform(dict(actor=delayed_adam(actor_rate,updates),
                                     critic=optax.adam(critic_rate)),labels)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--actor-warmup-steps",type=int,required=True)
    args,rest=parser.parse_known_args()
    total=int(rest[rest.index("--steps")+1])
    if not 0 < args.actor_warmup_steps < total or args.actor_warmup_steps % 2560:
        parser.error("Warmup must be a positive multiple of 2560 below total steps")
    if "--critic-lr" not in rest or "--teacher-kl" in rest:
        parser.error("Specify critic LR explicitly and do not confound with teacher KL")
    output=Path(rest[rest.index("--output")+1])
    updates=args.actor_warmup_steps//2560*16
    original_factory=constrained.split_adam
    original_train=constrained.improvement.runner.train.train
    constrained.split_adam=lambda actor,critic:warmup_optimizer(actor,critic,updates)

    def train(*values,**kwargs):
        expected=dict(num_envs=32,batch_size=32,num_minibatches=4,unroll_length=20,num_updates_per_batch=4)
        if any(kwargs[k]!=v for k,v in expected.items()):
            raise ValueError("Warmup update conversion assumes the frozen S09 rollout batch")
        snapshot=output/"sources"/Path(__file__).name
        (output/"warmup_config.json").write_text(json.dumps(dict(
            actor_warmup_transitions=args.actor_warmup_steps,actor_warmup_gradient_updates=updates,
            total_online_transitions=total,critic_warmup_is_included_in_budget=True,
            actor_moments_frozen=True,action_architecture_unchanged=True,
            source_sha256=digest(snapshot)),indent=2)+"\n")
        return original_train(*values,**kwargs)

    constrained.improvement.runner.train.train=train
    sys.argv=[sys.argv[0],*rest]
    try:
        constrained.main()
    finally:
        constrained.split_adam=original_factory
        constrained.improvement.runner.train.train=original_train


if __name__ == "__main__":
    main()
