"""Independent full-target controller: frozen acquisition, trainable feedback.

Two BC actors switch at the predeclared 2s boundary using elapsed observation
time. No reference action is added. Both actors are stored in the checkpoint;
the acquisition branch and its distribution parameters have zero gradient.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import pickle
import sys

from brax.training.networks import FeedForwardNetwork
import jax
import jax.numpy as jp
import numpy as np

import run_s09_constrained_ppo as constrained
from retarget_s09_reference import digest

runner = constrained.improvement.runner
BASE_FACTORY = runner.network_factory
BASE_INITIALIZE = runner.initialize_matched
ARCHITECTURE = "frozen acquisition BC before 2s, feedback full-target actor thereafter"


def network_factory(observation_size, action_size, **kwargs):
    base = BASE_FACTORY(observation_size, action_size, **kwargs)
    actor = base.policy_network

    def init(key):
        params = actor.init(key)
        return dict(acquisition=params, feedback=jax.tree_util.tree_map(jp.array, params))

    def apply(normalizer, params, observations):
        early = jax.lax.stop_gradient(actor.apply(normalizer, params["acquisition"], observations))
        late = actor.apply(normalizer, params["feedback"], observations)
        step = jp.rint(observations["state"][..., 112]*800).astype(jp.int32)
        return jp.where((step < 40)[..., None], early, late)

    return base.replace(policy_network=FeedForwardNetwork(init=init, apply=apply))


def initialize(env, seed, std):
    if env.arm != "pure":
        raise ValueError("Phase handoff supplies full targets, not residual actions")
    _, params = BASE_INITIALIZE(env, seed, std)
    combined = (params[0], dict(acquisition=params[1],
                              feedback=jax.tree_util.tree_map(jp.array, params[1])), params[2])
    return network_factory(dict(state=(123,), privileged_state=(140,)), 17), combined


@contextmanager
def handoff_networks():
    old = runner.network_factory, runner.initialize_matched
    runner.network_factory, runner.initialize_matched = network_factory, initialize
    try:
        yield
    finally:
        runner.network_factory, runner.initialize_matched = old


def read_checkpoint(path):
    parent = path.parent if (path.parent/"run_config.json").exists() else path.parent.parent
    return pickle.loads(path.read_bytes()), json.loads((parent/"run_config.json").read_text())


def combine(acquisition, feedback):
    if jax.tree_util.tree_structure(acquisition[0]) != jax.tree_util.tree_structure(feedback[0]):
        raise ValueError("Normalizer structures differ")
    for a, b in zip(jax.tree_util.tree_leaves(acquisition[0]), jax.tree_util.tree_leaves(feedback[0])):
        np.testing.assert_array_equal(a, b)
    if jax.tree_util.tree_structure(acquisition[1]) != jax.tree_util.tree_structure(feedback[1]):
        raise ValueError("Actor structures differ")
    return feedback[0], dict(acquisition=acquisition[1], feedback=feedback[1]), feedback[2]


def convert(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    first, cfg = read_checkpoint(args.acquisition)
    second, other = read_checkpoint(args.feedback)
    keys = ("model_sha256", "reference_sha256", "method", "physics_backend", "actor_dim", "critic_dim", "env_config")
    for key in keys:
        if cfg[key] != other[key]:
            raise ValueError("Incompatible source: "+key)
    if cfg["method"] != "guided" or cfg["env_config"].get("observation_variant") != "initial_pose_mode_onehot_95_101":
        raise ValueError("Use the original conditioned full-target policies")
    params = combine(first, second)
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output/"params.pkl"
    target.write_bytes(pickle.dumps(jax.device_get(params)))
    compatibility = {key: cfg[key] for key in keys}
    compatibility["policy_architecture"] = ARCHITECTURE
    (args.output/"run_config.json").write_text(json.dumps(compatibility, indent=2)+"\n")
    metadata = dict(status="combined_not_trained", acquisition=str(args.acquisition.resolve()),
        feedback=str(args.feedback.resolve()), acquisition_sha256=digest(args.acquisition),
        feedback_sha256=digest(args.feedback), checkpoint_sha256=digest(target),
        handoff_step=40, handoff_seconds=2, additional_training_updates=0,
        actor_parameter_multiplier=2, source_sha256=digest(Path(__file__)),
        limitation="Combining two pretrained BC actors is not new PPO learning or an action residual")
    (args.output/"conversion.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata), flush=True)


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "convert":
        return convert(argv[1:])
    if any(flag in argv for flag in ("--ppo-initial-std", "--teacher-kl", "--dataset")):
        raise ValueError("No unrelated actor interventions in the phase handoff experiment")
    if "--checkpoint" not in argv or argv[argv.index("--method")+1] != "guided":
        raise ValueError("Supply a converted full-target handoff checkpoint")
    checkpoint = Path(argv[argv.index("--checkpoint")+1])
    source, cfg = read_checkpoint(checkpoint)
    if cfg.get("policy_architecture") != ARCHITECTURE:
        raise ValueError("Convert explicitly before training or evaluation")
    output = Path(argv[argv.index("--output")+1])
    if output.exists():
        raise FileExistsError(output)
    original_train = runner.train.train

    def metadata():
        path = output/"run_config.json"
        if not path.exists():
            return
        current = json.loads(path.read_text())
        current["policy_architecture"] = ARCHITECTURE
        path.write_text(json.dumps(current, indent=2)+"\n")
        improvement = output/"improvement_config.json"
        if improvement.exists():
            options = json.loads(improvement.read_text())
            options["action_architecture"] = ARCHITECTURE
            improvement.write_text(json.dumps(options, indent=2)+"\n")
        counts = {key: sum(x.size for x in jax.tree_util.tree_leaves(source[1][key]))
                  for key in ("acquisition", "feedback")}
        record = dict(policy_architecture=ARCHITECTURE, routing="round(raw_state[112]*800) < 40",
            frozen_acquisition=True, actor_parameters=counts, reference_action_added=False,
            evaluation_uses_normal_reset=True, checkpoint_sha256=digest(checkpoint),
            source_sha256=digest(output/"sources"/Path(__file__).name),
            limitation="Two pretrained actors, not original single-network BC+PPO; switching may cause target discontinuity")
        (output/"phase_handoff_config.json").write_text(json.dumps(record, indent=2)+"\n")

    def train(*values, **kwargs):
        if kwargs.get("normalize_observations"):
            raise ValueError("Phase routing needs unchanged raw time observations")
        metadata()
        return original_train(*values, **kwargs)

    runner.train.train = train
    try:
        with handoff_networks():
            constrained.main()
    finally:
        runner.train.train = original_train
        metadata()


if __name__ == "__main__":
    main()
