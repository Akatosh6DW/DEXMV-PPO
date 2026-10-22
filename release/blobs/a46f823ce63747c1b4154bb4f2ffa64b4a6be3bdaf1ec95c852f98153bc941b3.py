"""Independent BC+PPO variant: six initial-pose-routed full-target actors.

Each expert starts as an exact BC parameter copy. Routing uses the existing
remembered pose label, never success labels or reference joint targets. This
increases actor parameter count sixfold and is not a residual controller.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import sys

import run_s09_constrained_ppo as constrained
from brax.training.networks import FeedForwardNetwork
import jax
import jax.numpy as jp

runner = constrained.improvement.runner
BASE_FACTORY = runner.network_factory
BASE_INITIALIZE = runner.initialize_matched
ARCHITECTURE = "six initial-pose-routed full-target actors"


def expand_parameters(params):
    if "experts" in params[1]:
        raise ValueError("Do not expand an already expanded actor")
    copies = tuple(jax.tree_util.tree_map(lambda x: jp.array(x), params[1]) for _ in range(6))
    return params[0], {"experts": copies}, params[2]


def network_factory(observation_size, action_size, **kwargs):
    base = BASE_FACTORY(observation_size, action_size, **kwargs)
    actor = base.policy_network

    def init(key):
        params = actor.init(key)
        return {"experts": tuple(jax.tree_util.tree_map(lambda x: jp.array(x), params)
                                 for _ in range(6))}

    def apply(normalizer, params, observations):
        if len(params["experts"]) != 6:
            raise ValueError("Exactly six pose experts are required")
        raw = observations["state"]
        # All actors see the full feedback. Only the selected actor supplies
        # distribution parameters, consistently for rollout and PPO likelihood.
        logits = jp.stack([actor.apply(normalizer, expert, observations)
                           for expert in params["experts"]], axis=-2)
        mode = jp.argmax(raw[..., 95:101], axis=-1)
        return jp.take_along_axis(logits, mode[..., None, None], axis=-2)[..., 0, :]

    return base.replace(policy_network=FeedForwardNetwork(init=init, apply=apply))


def initialize(env, seed, std):
    if env.arm != "pure":
        raise ValueError("Experts are full-target BC+PPO, not residual control")
    _, params = BASE_INITIALIZE(env, seed, std)
    net = network_factory(dict(state=(123,), privileged_state=(140,)), 17)
    return net, expand_parameters(params)


@contextmanager
def expert_networks():
    old_factory, old_initialize = runner.network_factory, runner.initialize_matched
    runner.network_factory, runner.initialize_matched = network_factory, initialize
    try:
        yield
    finally:
        runner.network_factory, runner.initialize_matched = old_factory, old_initialize


def main():
    argv = sys.argv[1:]
    if any(flag in argv for flag in ("--ppo-initial-std", "--teacher-kl", "--dataset")):
        raise ValueError("Do not combine expert routing with other actor interventions")
    if "--checkpoint" not in argv or argv[argv.index("--method")+1] != "guided":
        raise ValueError("Supply an explicitly expanded guided checkpoint")
    checkpoint = Path(argv[argv.index("--checkpoint")+1])
    parent = checkpoint.parent
    if not (parent/"run_config.json").exists():
        parent = parent.parent
    cfg = json.loads((parent/"run_config.json").read_text())
    if cfg.get("policy_architecture") != ARCHITECTURE:
        raise ValueError("Convert the BC checkpoint explicitly before expert training")
    output = Path(argv[argv.index("--output")+1])
    old_train = runner.train.train

    def metadata():
        path = output/"run_config.json"
        if path.exists():
            current = json.loads(path.read_text())
            current["policy_architecture"] = ARCHITECTURE
            path.write_text(json.dumps(current, indent=2)+"\n")
        path = output/"improvement_config.json"
        if path.exists():
            current = json.loads(path.read_text())
            current["action_architecture"] = ARCHITECTURE
            path.write_text(json.dumps(current, indent=2)+"\n")

    def train(*args, **kwargs):
        if kwargs.get("normalize_observations"):
            raise ValueError("Routing requires the unchanged raw mode one-hot")
        params = kwargs["restore_params"]
        counts = [sum(x.size for x in jax.tree_util.tree_leaves(expert))
                  for expert in params[1]["experts"]]
        record = dict(policy_architecture=ARCHITECTURE, routing="argmax(raw_state[95:101])",
            expert_count=6, actor_parameters_per_expert=counts,
            total_actor_parameters=sum(counts), critic_architecture_unchanged=True,
            reference_joint_targets_used=False, shared_actor_parameters=False,
            caveat="Adam momentum can update an expert with zero current minibatch gradient")
        (output/"mode_expert_config.json").write_text(json.dumps(record, indent=2)+"\n")
        metadata()
        return old_train(*args, **kwargs)

    runner.train.train = train
    try:
        with expert_networks():
            constrained.main()
    finally:
        runner.train.train = old_train
        metadata()


if __name__ == "__main__":
    main()
