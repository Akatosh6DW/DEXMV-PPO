"""Clone a BC actor into six pose experts without fitting any new data."""

import argparse
import json
from pathlib import Path
import pickle

from run_s09_mode_experts import ARCHITECTURE, expand_parameters
from retarget_s09_reference import digest
import jax
import numpy as np


def verify_copies(before, after):
    for left, right in ((before[0], after[0]), (before[2], after[2]),
                        *((before[1], x) for x in after[1]["experts"])):
        if jax.tree_util.tree_structure(left) != jax.tree_util.tree_structure(right):
            raise ValueError("Unexpected parameter structure")
        for x, y in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
            np.testing.assert_array_equal(x, y)


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
        parser.error("Use a pose-conditioned full-target BC checkpoint")
    original_hash = digest(args.source)
    before = pickle.loads(args.source.read_bytes())
    after = jax.device_get(expand_parameters(before))
    verify_copies(before, after)
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output/"params.pkl"
    destination.write_bytes(pickle.dumps(after))
    compatibility = {key:cfg[key] for key in (
        "model_sha256", "reference_sha256", "method", "physics_backend",
        "actor_dim", "critic_dim", "env_config")}
    compatibility.update(policy_architecture=ARCHITECTURE, source_checkpoint_sha256=original_hash)
    (args.output/"run_config.json").write_text(json.dumps(compatibility, indent=2)+"\n")
    report = dict(status="copied_not_trained", source_checkpoint=str(args.source.resolve()),
        source_checkpoint_sha256=original_hash, derived_checkpoint_sha256=digest(destination),
        all_six_actor_copies_bitwise_identical=True, critic_and_normalizer_unchanged=True,
        additional_demonstrations=0, actor_parameter_multiplier=6,
        source_sha256=digest(Path(__file__)))
    assert digest(args.source) == original_hash
    (args.output/"conversion.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
