"""Exclude student-generated prefix labels, retaining verified teacher suffixes.

The active mask here denotes supervised sample inclusion, not physical survival.
All original training and validation samples remain unchanged.
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from retarget_s09_reference import digest


def filter_prefix(arrays, train_count, recovery_count, prefix_steps):
    if not 0 < recovery_count < train_count < len(arrays["obs"]):
        raise ValueError("Require original training, added recovery, and validation episodes")
    if not 0 < prefix_steps < arrays["active"].shape[1]:
        raise ValueError("Prefix must leave a nonempty recovery suffix")
    result = {key: value.copy() for key, value in arrays.items()}
    result["active"][train_count-recovery_count:train_count, :prefix_steps] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-steps", type=int, default=60)
    args = parser.parse_args()
    manifest = json.loads((args.input/"dataset_manifest.json").read_text())
    source = args.input/"trajectories.npz"
    assert digest(source) == manifest["dataset_sha256"]
    assert manifest["original_validation_preserved"]
    with np.load(source) as z:
        arrays = dict(z)
    split, count = manifest["train_episodes"], manifest["additional_simulated_recovery_episodes"]
    filtered = filter_prefix(arrays, split, count, args.prefix_steps)
    for key in arrays:
        np.testing.assert_array_equal(filtered[key][:split-count], arrays[key][:split-count])
        np.testing.assert_array_equal(filtered[key][split:], arrays[key][split:])
        if key != "active":
            np.testing.assert_array_equal(filtered[key], arrays[key])
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/"trajectories.npz", **filtered)
    manifest.update(parent_dataset_sha256=digest(source), dataset_sha256=digest(args.output/"trajectories.npz"),
        excluded_student_prefix_steps=args.prefix_steps, sample_mask_semantics="supervised inclusion, not survival",
        source_sha256=digest(Path(__file__)))
    (args.output/"dataset_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    shutil.copyfile(Path(__file__), args.output/Path(__file__).name)
    report = dict(status="completed", excluded_samples=int(arrays["active"].sum()-filtered["active"].sum()),
        unchanged_original_training_episodes=split-count, unchanged_validation_episodes=len(arrays["obs"])-split,
        added_recovery_episodes=count, dataset_sha256=manifest["dataset_sha256"])
    (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
