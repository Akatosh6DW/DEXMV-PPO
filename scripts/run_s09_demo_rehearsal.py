"""Retain train-only demonstrations while refining the repaired BC with PPO.

The deployment actor still predicts full joint targets. Demonstrations enter
only the training objective, never the environment action interface.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import losses

import run_s09_constrained_ppo as constrained
from run_s09_bc_conditioned import condition_arrays
from run_s09_bc_repair import anchored_loss
from orca_s09_env import cylinder_axis
from retarget_s09_reference import digest


def load_rehearsal(dataset, inp):
    runner = constrained.improvement.runner
    runner.dataset_arrays(dataset, True, digest(inp / "model.xml"),
                          digest(inp / "reference.npz"), runner.OrcaS09Grasp.physics_backend)
    with np.load(inp / "reference.npz", allow_pickle=False) as archive:
        axes = np.asarray(jax.vmap(cylinder_axis)(jp.asarray(archive["object_quaternion"])))
    arrays = condition_arrays(dataset, axes)
    for obs in arrays[::2]:
        modes = np.asarray(obs[:, 95:101])
        if not np.all((modes == 0) | (modes == 1)) or not np.all(modes.sum(axis=1) == 1):
            raise ValueError("Rehearsal observations need remembered one-hot modes")
    # Validation samples are discarded before constructing the training loss.
    return arrays[:4]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--demo-dataset", type=Path, required=True)
    parser.add_argument("--demo-weight", type=float, default=10.)
    args, rest = parser.parse_known_args()
    if not math.isfinite(args.demo_weight) or args.demo_weight <= 0:
        parser.error("Use a finite positive rehearsal weight")
    if "--method" not in rest or rest[rest.index("--method") + 1] != "guided":
        parser.error("Rehearsal is a full-target BC+PPO intervention")
    if "--teacher-kl" in rest:
        parser.error("Do not confound the rehearsal test with online teacher KL")
    inp = Path(rest[rest.index("--input") + 1])
    output = Path(rest[rest.index("--output") + 1])
    arrays = load_rehearsal(args.demo_dataset, inp)
    runner = constrained.improvement.runner
    original_train, original_loss = runner.train.train, losses.compute_ppo_loss

    def train(*values, **kwargs):
        losses.compute_ppo_loss = anchored_loss(original_loss, arrays, args.demo_weight)
        metadata = dict(dataset=str(args.demo_dataset.resolve()),
            dataset_sha256=digest(args.demo_dataset), demo_weight=args.demo_weight,
            early_training_frames=len(arrays[0]), late_training_frames=len(arrays[2]),
            acquisition_steps=30, samples_per_update=256, early_late_samples=[128, 128],
            validation_used_for_training=False, full_feedback_inputs=True,
            conditioned_initial_pose=True, used_at_inference=False,
            source_sha256=digest(output / "sources" / Path(__file__).name))
        (output / "rehearsal_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return original_train(*values, **kwargs)

    runner.train.train = train
    sys.argv = [sys.argv[0], *rest]
    try:
        constrained.main()
    finally:
        runner.train.train = original_train
        losses.compute_ppo_loss = original_loss


if __name__ == "__main__":
    main()
