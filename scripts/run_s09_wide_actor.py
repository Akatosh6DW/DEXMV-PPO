"""Near-sixfold-capacity single actor with zero-output capacity expansion.

This control keeps the source policy function and critic, but trains one dense
actor for every pose. It is neither routed experts nor residual control.
"""

import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import pickle
import sys

import jax
import jax.numpy as jp
import numpy as np

import run_s09_constrained_ppo as constrained
from retarget_s09_reference import digest

runner = constrained.improvement.runner
BASE_INITIALIZE = runner.initialize_matched
WIDTH = 397
ARCHITECTURE = "single dense full-target actor 397x397 zero-output expansion"


def widen_actor(actor, width=WIDTH, seed=19411):
    result = copy.deepcopy(actor)
    layers = actor["params"]
    size = np.asarray(layers["hidden_0"]["bias"]).size
    if width < size or np.shape(layers["hidden_1"]["kernel"]) != (size, size):
        raise ValueError("Require two equally wide hidden layers and no shrinking")
    target = result["params"]
    keys = jax.random.split(jax.random.PRNGKey(seed), 2)
    init = jax.nn.initializers.lecun_uniform()
    dtype = layers["hidden_0"]["kernel"].dtype
    first = init(keys[0], (layers["hidden_0"]["kernel"].shape[0], width), dtype)
    target["hidden_0"]["kernel"] = first.at[:, :size].set(layers["hidden_0"]["kernel"])
    target["hidden_0"]["bias"] = jp.zeros(width, dtype).at[:size].set(layers["hidden_0"]["bias"])
    # New units have random incoming weights, but cannot affect the old path
    # until PPO learns nonzero outgoing weights. They are not frozen.
    middle = init(keys[1], (width, width), dtype).at[:, :size].set(0)
    target["hidden_1"]["kernel"] = middle.at[:size, :size].set(layers["hidden_1"]["kernel"])
    target["hidden_1"]["bias"] = jp.zeros(width, dtype).at[:size].set(layers["hidden_1"]["bias"])
    final = jp.zeros((width, layers["hidden_2"]["kernel"].shape[1]), dtype)
    target["hidden_2"]["kernel"] = final.at[:size].set(layers["hidden_2"]["kernel"])
    return result, dict(old_width=size, new_width=width, seed=seed, new_outgoing_weights_zero=True)


def network_factory(observation_size, action_size, **kwargs):
    return runner.networks.make_ppo_networks(observation_size, action_size, **kwargs,
        policy_hidden_layer_sizes=(WIDTH, WIDTH), value_hidden_layer_sizes=(128, 128),
        policy_obs_key="state", value_obs_key="privileged_state", init_noise_std=.2,
        mean_kernel_init_fn=jax.nn.initializers.lecun_uniform)


def initialize(env, seed, std):
    if env.arm != "pure":
        raise ValueError("Wide control only supports full-target policies")
    _, params = BASE_INITIALIZE(env, seed, std)
    policy, _ = widen_actor(params[1])
    return network_factory(dict(state=(123,), privileged_state=(140,)), 17), (params[0], policy, params[2])


@contextmanager
def wide_networks():
    old_factory, old_initialize = runner.network_factory, runner.initialize_matched
    runner.network_factory, runner.initialize_matched = network_factory, initialize
    try:
        yield
    finally:
        runner.network_factory, runner.initialize_matched = old_factory, old_initialize


def convert(source, output):
    jax.config.update("jax_default_matmul_precision", "highest")
    parent = source.parent if (source.parent/"run_config.json").exists() else source.parent.parent
    cfg = json.loads((parent/"run_config.json").read_text())
    if cfg["method"] != "guided" or cfg.get("policy_architecture") or cfg["env_config"].get("angular_velocity_observation"):
        raise ValueError("Use the original single full-target policy, not another intervention")
    if cfg["env_config"].get("observation_variant") != "initial_pose_mode_onehot_95_101":
        raise ValueError("Require the same pose-conditioned observation")
    before = pickle.loads(source.read_bytes())
    actor, expansion = widen_actor(before[1])
    after = (before[0], actor, before[2])
    env = constrained.improvement.ModeConditionedGrasp(Path(cfg["input"])/"model.xml", "pure")
    base, _ = BASE_INITIALIZE(env, cfg["seed"], cfg["std"])
    wide = network_factory(dict(state=(123,), privileged_state=(140,)), 17)
    rng = np.random.default_rng(19412)
    observations = [rng.normal(size=(64, 123)).astype(np.float32)]
    preview = source.parent/"preview_trajectories.npz"
    if preview.exists():
        with np.load(preview) as z:
            observations.append(z["obs"][z["active"]])
    errors = []
    for obs in observations:
        inputs = dict(state=jp.asarray(obs))
        a = np.asarray(base.policy_network.apply(before[0], before[1], inputs))
        b = np.asarray(wide.policy_network.apply(after[0], after[1], inputs))
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=0)
        errors.append(float(np.max(np.abs(a-b))))
    count = lambda tree: sum(x.size for x in jax.tree_util.tree_leaves(tree))
    output.mkdir(parents=True, exist_ok=False)
    path = output/"params.pkl"
    path.write_bytes(pickle.dumps(jax.device_get(after)))
    cfg.update(policy_architecture=ARCHITECTURE, output=str(output.resolve()))
    (output/"run_config.json").write_text(json.dumps(cfg, indent=2)+"\n")
    record = dict(status="expanded_not_trained", source_checkpoint=str(source.resolve()),
        source_checkpoint_sha256=digest(source), derived_checkpoint_sha256=digest(path),
        source_actor_parameters=count(before[1]), wide_actor_parameters=count(actor),
        six_expert_actor_parameters=6*count(before[1]), expansion_seed=19411,
        expansion=expansion, compared_states=[len(x) for x in observations], max_logit_errors=errors,
        critic_and_normalizer_unchanged=True, extra_online_steps=0,
        caveat="Function preserving in real arithmetic; float32 physics may diverge in contact-sensitive trajectories",
        source_sha256=digest(Path(__file__)))
    (output/"conversion.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--convert-source", type=Path)
    parser.add_argument("--conversion-output", type=Path)
    args, rest = parser.parse_known_args()
    if args.convert_source:
        if not args.conversion_output or rest:
            parser.error("Conversion requires source and output only")
        convert(args.convert_source, args.conversion_output)
        return
    if any(flag in rest for flag in ("--teacher-kl", "--ppo-initial-std", "--dataset")):
        parser.error("Keep the capacity intervention separate from other actor changes")
    if rest[rest.index("--method")+1] != "guided":
        parser.error("Only full-target BC+PPO is supported")
    source = Path(rest[rest.index("--checkpoint")+1])
    parent = source.parent if (source.parent/"run_config.json").exists() else source.parent.parent
    cfg = json.loads((parent/"run_config.json").read_text())
    if cfg.get("policy_architecture") != ARCHITECTURE:
        parser.error("Convert the checkpoint before training")
    output = Path(rest[rest.index("--output")+1])
    old_train = runner.train.train

    def metadata():
        path = output/"run_config.json"
        if path.exists():
            data = json.loads(path.read_text())
            data["policy_architecture"] = ARCHITECTURE
            path.write_text(json.dumps(data, indent=2)+"\n")

    def train(*values, **kwargs):
        metadata()
        params = kwargs["restore_params"]
        counts = {key:sum(x.size for x in jax.tree_util.tree_leaves(params[i]))
                  for key,i in (("actor",1),("critic",2))}
        (output/"wide_actor_config.json").write_text(json.dumps(dict(
            policy_architecture=ARCHITECTURE, parameter_counts=counts,
            pose_routing=False, critic_architecture_unchanged=True), indent=2)+"\n")
        return old_train(*values, **kwargs)

    runner.train.train = train
    sys.argv = [sys.argv[0], *rest]
    try:
        with wide_networks():
            constrained.main()
    finally:
        runner.train.train = old_train
        metadata()


if __name__ == "__main__":
    main()
