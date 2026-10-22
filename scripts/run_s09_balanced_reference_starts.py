"""Class-balanced reference resets; the original evaluation stays unchanged."""

import json
import math
from pathlib import Path
import sys

import jax
import numpy as np

import run_s09_reference_starts as reference_starts
from retarget_s09_reference import digest


def balanced_indices(modes):
    modes = np.asarray(modes)
    if modes.ndim != 1 or not len(modes) or not np.isin(modes, np.arange(6)).all():
        raise ValueError("Expected a nonempty vector of six-class indices")
    counts = np.bincount(modes.astype(int), minlength=6)
    present = np.flatnonzero(counts)
    per_class = math.lcm(*(int(counts[k]) for k in present))
    if per_class*len(present) > 10000:
        raise ValueError("Exact balancing would expand the bank excessively")
    # Duplicate existing states to give every present class equal probability,
    # and every original state within that class equal probability.
    indices = np.concatenate([np.tile(np.flatnonzero(modes == k), per_class//counts[k])
                              for k in present])
    return indices.astype(np.int32), counts


def main():
    output = Path(sys.argv[sys.argv.index("--output")+1])
    original = reference_starts.ReferenceStarts

    def make_balanced(env, bank, probability):
        indices, counts = balanced_indices(bank.info["reference_index"])
        expanded = jax.tree_util.tree_map(lambda x: x[indices], bank)
        wrapped = original(env, expanded, probability)
        present = counts > 0
        metadata = dict(original_count=len(bank.reward), original_class_counts=counts.tolist(),
            expanded_count=len(indices), expanded_to_original=indices.tolist(),
            conditional_reference_class_probability=(present/present.sum()).tolist(),
            reset_probability=probability, no_new_physical_states=True,
            collection_interactions_added=0, evaluation_wrapped=False,
            source_sha256=digest(output/"sources"/Path(__file__).name),
            parent_source_sha256=digest(output/"sources"/"run_s09_reference_starts.py"),
            limitation="Only existing classes are balanced; absent classes remain absent. Repetition is not new teacher data.")
        (output/"balanced_reset_config.json").write_text(json.dumps(metadata, indent=2)+"\n")
        return wrapped

    reference_starts.ReferenceStarts = make_balanced
    try:
        reference_starts.main()
    finally:
        reference_starts.ReferenceStarts = original


if __name__ == "__main__":
    main()
