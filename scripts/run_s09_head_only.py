"""Keep BC hidden features fixed while PPO fine-tunes its output layer.

Both mean and scale outputs remain trainable. This is the original single
full-target actor, not a frozen BC action plus another controller's correction.
"""

import json
from pathlib import Path
import pickle
import sys

import jax
import jax.numpy as jp
import numpy as np
import optax

import run_s09_constrained_ppo as constrained
from retarget_s09_reference import digest


def is_output(path):
    names = tuple(getattr(key, "key", None) for key in path)
    return names in (("params", "hidden_2", "kernel"), ("params", "hidden_2", "bias"))


def head_only_gradients():
    def init(params):
        if set(params.policy["params"]) != {"hidden_0", "hidden_1", "hidden_2"}:
            raise ValueError("Expected the original three-layer actor")
        if params.policy["params"]["hidden_2"]["bias"].shape != (34,):
            raise ValueError("Expected 17 mean and 17 scale outputs")
        return optax.EmptyState()

    def update(grads, state, params=None):
        del params
        policy = jax.tree_util.tree_map_with_path(
            lambda path, value: value if is_output(path) else jp.zeros_like(value), grads.policy)
        return grads.replace(policy=policy), state

    return optax.GradientTransformation(init, update)


def verify_frozen_hidden(before, after):
    left, tree = jax.tree_util.tree_flatten_with_path(before[1])
    right, other_tree = jax.tree_util.tree_flatten_with_path(after[1])
    if tree != other_tree:
        raise ValueError("Actor structure changed")
    for (path, x), (_, y) in zip(left, right):
        if not is_output(path):
            np.testing.assert_array_equal(x, y)


def main():
    if "--critic-lr" not in sys.argv or sys.argv[sys.argv.index("--method")+1] != "guided":
        raise ValueError("Use explicit split rates and the full-target actor")
    if any(option in sys.argv for option in ("--teacher-kl", "--ppo-discount", "--ppo-initial-std")):
        raise ValueError("Keep this parameter-scope probe separate from other interventions")
    output = Path(sys.argv[sys.argv.index("--output")+1])
    runner = constrained.improvement.runner
    original_split, original_train = constrained.split_adam, runner.train.train

    def split(actor_rate, critic_rate):
        # Fresh Adam states start at zero; masking before Adam prevents both
        # hidden-layer gradients and hidden-layer optimizer moments changing.
        return optax.chain(head_only_gradients(), original_split(actor_rate, critic_rate))

    def train(*values, **kwargs):
        head = kwargs["restore_params"][1]["params"]["hidden_2"]
        metadata = dict(actor_update_rule="output_layer_mean_and_scale_only",
            trainable_actor_parameters=sum(np.size(x) for x in head.values()),
            hidden_features_frozen=True, critic_updates_changed=False,
            inference_changed=False, reference_action_added=False,
            reward_version="S09-R0 unchanged",
            source_sha256=digest(output/"sources"/Path(__file__).name))
        (output/"head_only_config.json").write_text(json.dumps(metadata, indent=2)+"\n")
        return original_train(*values, **kwargs)

    constrained.split_adam, runner.train.train = split, train
    try:
        constrained.main()
    finally:
        constrained.split_adam, runner.train.train = original_split, original_train
        if (output/"summary.json").exists():
            first, last = output/"step_000000000/params.pkl", output/"params.pkl"
            before, after = pickle.loads(first.read_bytes()), pickle.loads(last.read_bytes())
            verify_frozen_hidden(before, after)
            old_head, new_head = before[1]["params"]["hidden_2"], after[1]["params"]["hidden_2"]
            changed = sum(int(np.count_nonzero(np.asarray(old_head[k]) != np.asarray(new_head[k]))) for k in old_head)
            audit = dict(status="verified", hidden_parameters_bitwise_unchanged=True,
                changed_output_parameters=changed, initial_checkpoint_sha256=digest(first),
                final_checkpoint_sha256=digest(last))
            (output/"head_only_audit.json").write_text(json.dumps(audit, indent=2)+"\n")


if __name__ == "__main__":
    main()
