"""PPO updates only the BC actor's previously masked feedback input weights.

All inference inputs remain present and a single network emits full targets.
This is restricted parameter fine-tuning, not additive residual control.
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
from run_s09_bc_phase_curriculum import KEEP
from retarget_s09_reference import digest


def is_first_kernel(path):
    return tuple(getattr(key, "key", None) for key in path) == ("params", "hidden_0", "kernel")


def feedback_only_gradients():
    def init(params):
        if params.policy["params"]["hidden_0"]["kernel"].shape[0] != len(KEEP):
            raise ValueError("Expected the repaired 123-input actor")
        return optax.EmptyState()

    def update(grads, state, params=None):
        del params
        def mask(path, value):
            return value * jp.asarray(~KEEP)[:, None] if is_first_kernel(path) else jp.zeros_like(value)
        policy = jax.tree_util.tree_map_with_path(mask, grads.policy)
        return grads.replace(policy=policy), state

    return optax.GradientTransformation(init, update)


def verify_frozen_actor(before, after):
    left, tree = jax.tree_util.tree_flatten_with_path(before[1])
    right, other_tree = jax.tree_util.tree_flatten_with_path(after[1])
    if tree != other_tree:
        raise ValueError("Actor structure changed")
    for (path, x), (_, y) in zip(left, right):
        x, y = np.asarray(x), np.asarray(y)
        if is_first_kernel(path):
            np.testing.assert_array_equal(x[KEEP], y[KEEP])
        else:
            np.testing.assert_array_equal(x, y)


def main():
    if "--critic-lr" not in sys.argv or sys.argv[sys.argv.index("--method")+1] != "guided":
        raise ValueError("Use explicit split rates and a full-target guided actor")
    if "--teacher-kl" in sys.argv:
        raise ValueError("Do not confound the parameter-freezing comparison with KL anchoring")
    output = Path(sys.argv[sys.argv.index("--output")+1])
    original_split = constrained.split_adam
    runner = constrained.improvement.runner
    original_train = runner.train.train

    def split(actor_rate, critic_rate):
        # Mask before Adam: frozen coordinates keep zero moments and updates.
        return optax.chain(feedback_only_gradients(), original_split(actor_rate, critic_rate))

    def train(*values, **kwargs):
        width = kwargs["restore_params"][1]["params"]["hidden_0"]["kernel"].shape[1]
        metadata = dict(actor_update_rule="first_layer_non_phase_feedback_rows_only",
            trainable_input_rows=np.flatnonzero(~KEEP).tolist(),
            trainable_actor_parameter_count=int((~KEEP).sum() * width),
            bc_backbone_frozen=True, std_head_frozen=True, critic_updates_changed=False,
            inference_changed=False, reference_action_added=False,
            reward_version="S09-R0 unchanged",
            source_sha256=digest(output / "sources" / Path(__file__).name))
        (output / "adapter_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return original_train(*values, **kwargs)

    constrained.split_adam = split
    runner.train.train = train
    try:
        constrained.main()
    finally:
        constrained.split_adam = original_split
        runner.train.train = original_train
        if (output / "summary.json").exists():
            before_path = output / "step_000000000/params.pkl"
            after_path = output / "params.pkl"
            before, after = pickle.loads(before_path.read_bytes()), pickle.loads(after_path.read_bytes())
            verify_frozen_actor(before, after)
            first = lambda p:np.asarray(p[1]["params"]["hidden_0"]["kernel"])[~KEEP]
            audit = dict(status="verified", frozen_actor_parameters_exactly_unchanged=True,
                changed_feedback_weights=int(np.count_nonzero(first(before) != first(after))),
                initial_checkpoint_sha256=digest(before_path), final_checkpoint_sha256=digest(after_path))
            (output / "adapter_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
