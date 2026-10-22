"""Learn only new feedback input connections in copies of a frozen phase BC.

Old weights stay exact, but new feedback can still change the executed actions.
This is a parameter-freezing ablation, not an output residual or Stable-BC.
"""

import json
from pathlib import Path
import sys

import jax
import jax.numpy as jp
import numpy as np
import optax

import run_s09_recovery_expert_bc as expert_bc
from retarget_s09_reference import digest
from run_s09_bc_phase_curriculum import KEEP


def feedback_only(updates):
    def project(path, value):
        keys = tuple(getattr(k, "key", getattr(k, "idx", None)) for k in path)
        if len(keys) == 5 and keys[0] == "experts" and keys[2:] == ("params", "hidden_0", "kernel"):
            if value.shape[0] != len(KEEP):
                raise ValueError("Unexpected feedback input dimension")
            return value * jp.asarray(~KEEP)[:, None]
        return jp.zeros_like(value)
    return jax.tree_util.tree_map_with_path(project, updates)


class AdapterOptimizer:
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def adam(self, learning_rate):
        def update(updates, state, params=None):
            return feedback_only(updates), state
        projection = optax.GradientTransformation(lambda params: optax.EmptyState(), update)
        # Project Adam's actual step, not just its gradient, to enforce freezing.
        return optax.chain(self.original.adam(learning_rate), projection)


def verify_frozen(before, after):
    differences = jax.tree_util.tree_map(lambda a, b: np.asarray(b) - np.asarray(a), before, after)
    projected = feedback_only(differences)
    for difference, allowed in zip(jax.tree_util.tree_leaves(differences), jax.tree_util.tree_leaves(projected)):
        np.testing.assert_array_equal(difference, allowed)
    return sum(int(np.count_nonzero(x)) for x in jax.tree_util.tree_leaves(differences))


def main():
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    original_optimizer = expert_bc.optax
    original_expand_fit = expert_bc.fit_expanded

    def fit(original, net, params, arrays, updates, seed, optimizer_module):
        result, losses = original_expand_fit(original, net, params, arrays, updates, seed, optimizer_module)
        before = expert_bc.experts.expand_parameters(params)
        changed = verify_frozen(before[1], result[1])
        for index in (0, 2):
            for a, b in zip(jax.tree_util.tree_leaves(before[index]), jax.tree_util.tree_leaves(result[index])):
                np.testing.assert_array_equal(a, b)
        record = dict(status="verified_parameter_scope", trainable="experts/*/params/hidden_0/kernel[~KEEP]",
            changed_parameters=changed, bc_updates=updates, ppo_steps=0,
            old_connections_frozen=True, critic_normalizer_unchanged=True,
            output_actions_not_guaranteed_unchanged=True, source_sha256=digest(Path(__file__)))
        (output / "feedback_adapter_config.json").write_text(json.dumps(record, indent=2) + "\n")
        return result, losses

    expert_bc.optax = AdapterOptimizer(original_optimizer)
    expert_bc.fit_expanded = fit
    try:
        expert_bc.main()
    finally:
        expert_bc.fit_expanded = original_expand_fit
        expert_bc.optax = original_optimizer


if __name__ == "__main__":
    main()
