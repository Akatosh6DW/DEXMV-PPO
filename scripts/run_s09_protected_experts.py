"""Fine-tune a checkpoint portfolio while preserving its earlier-stage actors.

Frozen pose categories come only from the prior development selection. This
changes trainable PPO parameters, not task resets, rewards, or action routing.
"""

import json
from pathlib import Path
import pickle
import sys

import jax
import jax.numpy as jp
import numpy as np
import optax

import run_s09_mode_experts as experts
from retarget_s09_reference import digest


def protect_gradients(indices):
    frozen = tuple(indices)
    if not frozen or len(set(frozen)) != len(frozen) or any(i not in range(6) for i in frozen):
        raise ValueError("Require distinct protected expert indices in 0..5")

    def init(params):
        if len(params.policy["experts"]) != 6:
            raise ValueError("Require six routed actors")
        return optax.EmptyState()

    def update(grads, state, params=None):
        del params
        policy = dict(grads.policy)
        policy["experts"] = tuple(jax.tree_util.tree_map(jp.zeros_like, p) if i in frozen else p
                                  for i,p in enumerate(grads.policy["experts"]))
        return grads.replace(policy=policy), state

    return optax.GradientTransformation(init,update)


def audit_frozen(before, after, indices):
    rows = []
    for i,(a,b) in enumerate(zip(before[1]["experts"],after[1]["experts"])):
        assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
        changes = 0
        for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
            x,y = np.asarray(x),np.asarray(y)
            if i in indices:
                np.testing.assert_array_equal(x,y)
            changes += int(np.count_nonzero(x != y))
        rows.append(dict(mode=i, protected=i in indices, changed_parameters=changes))
    return rows


def main():
    argv = sys.argv[1:]
    if "--critic-lr" not in argv:
        raise ValueError("Use explicit split actor/critic rates")
    checkpoint = Path(argv[argv.index("--checkpoint")+1])
    selection_file = checkpoint.parent.parent/"selection.json"
    selection = json.loads(selection_file.read_text())
    if selection["status"] != "assembled_not_newly_trained":
        raise ValueError("Start from the fixed, untrained checkpoint portfolio")
    assert digest(checkpoint) == selection["candidate_sha256"]
    frozen = tuple(i for i,source in enumerate(selection["selected_source_by_mode"]) if source != "expert")
    output = Path(argv[argv.index("--output")+1])
    constrained = experts.constrained
    original_split = constrained.split_adam
    original_train = experts.runner.train.train

    def split(actor_rate, critic_rate):
        # This entry starts new Adam moments. Zero gradients before Adam keep
        # protected parameters fixed, including their distribution scales.
        return optax.chain(protect_gradients(frozen),original_split(actor_rate,critic_rate))

    def train(*args,**kwargs):
        record = dict(protected_modes=frozen, rule="freeze actors selected from BC/early stage; train endpoint actors",
            selection_sha256=digest(selection_file), source_checkpoint_sha256=digest(checkpoint),
            reward_unchanged=True, observation_unchanged=True, all_orientation_training=True,
            new_adam_moments=True, source_sha256=digest(Path(__file__)))
        (output/"protection_config.json").write_text(json.dumps(record,indent=2)+"\n")
        return original_train(*args,**kwargs)

    constrained.split_adam = split
    experts.runner.train.train = train
    try:
        experts.main()
    finally:
        constrained.split_adam = original_split
        experts.runner.train.train = original_train
        if (output/"summary.json").exists():
            before_path, after_path = output/"step_000000000/params.pkl", output/"params.pkl"
            before,after = [pickle.loads(p.read_bytes()) for p in (before_path,after_path)]
            record = dict(status="verified", experts=audit_frozen(before,after,frozen),
                initial_checkpoint_sha256=digest(before_path),final_checkpoint_sha256=digest(after_path),
                scope="Parameter retention only; physical success must be checked separately")
            (output/"protection_audit.json").write_text(json.dumps(record,indent=2)+"\n")
            print(json.dumps(record),flush=True)


if __name__ == "__main__":
    main()
