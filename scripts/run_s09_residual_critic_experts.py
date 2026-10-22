"""Matched shared/pose-routed critics for the six-actor learned-base residual."""

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

ARCHITECTURE = experts.ARCHITECTURE + " and six pose-routed value critics"


@contextmanager
def critic_experts(count):
    if count not in (1, 6):
        raise ValueError("Use one shared critic or six pose-routed critics")
    if count == 1:
        yield
        return
    previous = experts.residual_expert_factory, experts.SINGLE_INITIALIZE, experts.ARCHITECTURE

    def factory(base_action):
        make = previous[0](base_action)

        def network(*args, **kwargs):
            net = make(*args, **kwargs)
            critic = net.value_network

            def init(key):
                params = critic.init(key)
                return {"experts": tuple(jax.tree_util.tree_map(jp.array, params) for _ in range(6))}

            def apply(normalizer, params, observations):
                if len(params["experts"]) != 6:
                    raise ValueError("Six value critics required")
                values = jp.stack([critic.apply(normalizer, p, observations)
                                   for p in params["experts"]], axis=-1)
                mode = jp.argmax(observations["privileged_state"][..., 95:101], axis=-1)
                return jp.take_along_axis(values, mode[..., None], axis=-1)[..., 0]

            return net.replace(value_network=FeedForwardNetwork(init=init, apply=apply))
        return network

    def initialize(*args, **kwargs):
        net, params = previous[1](*args, **kwargs)
        value = {"experts": tuple(jax.tree_util.tree_map(jp.array, params[2]) for _ in range(6))}
        return net, (params[0], params[1], value)

    experts.residual_expert_factory, experts.SINGLE_INITIALIZE, experts.ARCHITECTURE = factory, initialize, ARCHITECTURE
    try:
        yield
    finally:
        experts.residual_expert_factory, experts.SINGLE_INITIALIZE, experts.ARCHITECTURE = previous


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--critic-experts", type=int, choices=(1, 6), required=True)
    args, rest = parser.parse_known_args()
    out = Path(rest[rest.index("--output") + 1])
    if "--checkpoint" in rest:
        parser.error("This matched probe starts from the frozen base, not a residual checkpoint")
    runner = experts.learned.runner
    old_train, old_argv = runner.train.train, sys.argv
    record = dict(critic_count=args.critic_experts, source_sha256=digest(Path(__file__)),
                  routing="argmax(raw privileged_state[95:101]); remembered initial pose",
                  unchanged=["six residual actors", "frozen base", "R0 reward", "action composition"],
                  limitation="Sixfold critic capacity and routing change together; not capacity-matched",
                  shared_loss_reduction=True, per_expert_learning_rate_rescaling=False)

    def train(*values, **kwargs):
        params = kwargs["restore_params"][2]
        critics = params["experts"] if args.critic_experts == 6 else (params,)
        record["parameters_per_critic"] = [sum(x.size for x in jax.tree_util.tree_leaves(p)) for p in critics]
        record["initial_critics_identical"] = all(
            all(np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(critics[0]),
                                                    jax.tree_util.tree_leaves(p))) for p in critics)
        if not record["initial_critics_identical"]:
            raise ValueError("Initial critics must be exact copies")
        (out / "critic_expert_config.json").write_text(json.dumps(record, indent=2) + "\n")
        return old_train(*values, **kwargs)

    runner.train.train, sys.argv = train, [sys.argv[0], *rest]
    try:
        with critic_experts(args.critic_experts):
            experts.main()
    finally:
        runner.train.train, sys.argv = old_train, old_argv
        if out.exists():
            (out / "critic_expert_config.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
