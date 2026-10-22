"""Create a diagnostic copy with post-BC feedback input weights zeroed.

Do not alter the original checkpoint or claim this intervention is training.
All other actor weights and the critic remain bitwise unchanged.
"""

import argparse
import json
from pathlib import Path
import pickle

import jax
import numpy as np

from run_s09_bc_phase_curriculum import KEEP, phase_initialization
from retarget_s09_reference import digest


def verify_only_feedback_removed(before, after):
    left, structure = jax.tree_util.tree_flatten_with_path(before)
    right, other = jax.tree_util.tree_flatten_with_path(after)
    if structure != other:
        raise ValueError("Checkpoint structure changed")
    changed = 0
    for (path, old), (_, new) in zip(left, right):
        keys = tuple(getattr(key, "key", getattr(key, "idx", None)) for key in path)
        old, new = np.asarray(old), np.asarray(new)
        if keys == (1, "params", "hidden_0", "kernel"):
            np.testing.assert_array_equal(old[KEEP], new[KEEP])
            np.testing.assert_array_equal(new[~KEEP], 0)
            changed = int(np.count_nonzero(old[~KEEP] != new[~KEEP]))
        else:
            np.testing.assert_array_equal(old, new)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent = args.source.parent
    if not (parent/"run_config.json").exists():
        parent = parent.parent
    cfg = json.loads((parent/"run_config.json").read_text())
    if cfg["method"] != "guided" or cfg["env_config"].get("observation_variant") != "initial_pose_mode_onehot_95_101":
        parser.error("Use the repaired full-target actor")
    original_hash = digest(args.source)
    before = pickle.loads(args.source.read_bytes())
    after = jax.device_get(phase_initialization(before))
    changed = verify_only_feedback_removed(before, after)
    if not changed:
        parser.error("No feedback weights changed; this would not test the ablation")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output/"params.pkl"
    checkpoint.write_bytes(pickle.dumps(after))
    # Minimal compatibility metadata, not a fabricated training run/config.
    compatibility = {key:cfg[key] for key in (
        "model_sha256", "reference_sha256", "method", "physics_backend",
        "actor_dim", "critic_dim", "env_config")}
    compatibility.update(diagnostic_only=True, source_checkpoint=str(args.source.resolve()),
                         source_checkpoint_sha256=original_hash)
    (args.output/"run_config.json").write_text(json.dumps(compatibility, indent=2)+"\n")
    report = dict(status="derived_not_trained", changed_feedback_weights=changed,
        kept_actor_rows=np.flatnonzero(KEEP).tolist(), other_parameters_exactly_unchanged=True,
        source_checkpoint=str(args.source.resolve()), source_checkpoint_sha256=original_hash,
        derived_checkpoint_sha256=digest(checkpoint), source_sha256=digest(Path(__file__)))
    assert digest(args.source) == original_hash
    (args.output/"ablation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
