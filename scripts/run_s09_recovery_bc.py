"""Controlled BC refinement: feedback inputs and verified recovery examples.

The default variants start from phase-only BC. An explicit feedback-source flag
supports a separate correction round from an already feedback-trained BC actor.
This entry point does not run PPO or add a reference at execution.
"""

import argparse
import json
from pathlib import Path
import pickle
import sys

import jax
import jax.numpy as jp
import numpy as np

import run_s09_bc_conditioned as conditioned
from run_s09_bc_phase_curriculum import KEEP
from retarget_s09_reference import digest
from orca_s09_env import cylinder_axis


def mask_arrays(arrays, phase_only):
    return tuple(x*jp.asarray(KEEP) if phase_only and i%2==0 else x for i,x in enumerate(arrays))


class BCOptimizerProxy:
    def __init__(self, original, rate):
        self.original, self.rate = original, rate

    def __getattr__(self, name):
        return getattr(self.original, name)

    def adam(self, learning_rate):
        return self.original.adam(self.rate)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bc-source", type=Path, required=True)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-4)
    parser.add_argument("--phase-only", action="store_true")
    parser.add_argument("--feedback-source", action="store_true",
                        help="Explicit independent correction round from an already feedback-trained flat actor")
    args, rest = parser.parse_known_args()
    if not 0 < args.bc_learning_rate < .01:
        parser.error("Invalid BC learning rate")
    if args.feedback_source and args.phase_only:
        parser.error("Do not silently remove feedback from an already trained feedback source")
    if "--checkpoint" in rest or "--repair-demo-weight" in rest:
        parser.error("No PPO continuation or auxiliary loss in this BC-only comparison")
    if rest[rest.index("--steps")+1] != "0" or rest[rest.index("--method")+1] != "guided":
        parser.error("Use guided --steps 0; PPO is evaluated separately")
    inp = Path(rest[rest.index("--input")+1])
    output = Path(rest[rest.index("--output")+1])
    dataset = Path(rest[rest.index("--dataset")+1])
    with np.load(inp/"reference.npz") as z:
        axes = np.asarray(jax.vmap(cylinder_axis)(jp.asarray(z["object_quaternion"], jp.float32)))
    cfg = json.loads((args.bc_source.parent.parent/"run_config.json").read_text())
    for key, file in (("model_sha256", "model.xml"), ("reference_sha256", "reference.npz")):
        if cfg[key] != digest(inp/file):
            parser.error("Source checkpoint and task assets differ")
    if cfg["method"] != "guided" or cfg["env_config"].get("observation_variant") != "initial_pose_mode_onehot_95_101":
        parser.error("Use the matching phase/mode full-target BC")
    if args.feedback_source and cfg.get("steps") != 0:
        parser.error("The correction round must start from BC-only, not a PPO-trained source")
    source = pickle.loads(args.bc_source.read_bytes())
    kernel = np.asarray(source[1]["params"]["hidden_0"]["kernel"])
    if kernel.shape[0] != 123:
        parser.error("Use the original 123-feature flat actor")
    if not args.feedback_source:
        np.testing.assert_array_equal(kernel[~KEEP], 0)
    runner = conditioned.repair.runner
    original_fit = runner.fit_matched_bc
    original_optax = original_fit.__globals__["optax"]
    source_hash = digest(args.bc_source)

    def fit(net, unused_initialization, arrays, updates, seed):
        # The outer repair wrapper reloads raw arrays for an explicit split.
        # Reapply mode conditioning after that reload, preserving the exact split.
        arrays = conditioned.condition_arrays(dataset, axes)
        metadata = dict(source_checkpoint=str(args.bc_source.resolve()), source_checkpoint_sha256=source_hash,
            phase_only=args.phase_only, bc_learning_rate=args.bc_learning_rate, bc_updates=updates,
            feedback_source_allowed=args.feedback_source,
            source_feedback_weights_nonzero=int(np.count_nonzero(kernel[~KEEP])),
            ppo_steps=0, reference_added_at_inference=False,
            explicit_split_reconditioned_after_repair=True,
            input_columns=np.flatnonzero(KEEP).tolist() if args.phase_only else list(range(123)),
            limitation="Small supervised refinement experiment, not evidence of PPO gain or a full DAgger implementation")
        (output/"recovery_bc_config.json").write_text(json.dumps(metadata, indent=2)+"\n")
        original_fit.__globals__["optax"] = BCOptimizerProxy(original_optax, args.bc_learning_rate)
        try:
            fitted, losses = original_fit(net, source, mask_arrays(arrays, args.phase_only), updates, seed)
        finally:
            original_fit.__globals__["optax"] = original_optax
        if args.phase_only:
            np.testing.assert_array_equal(np.asarray(fitted[1]["params"]["hidden_0"]["kernel"])[~KEEP], 0)
        return fitted, losses

    runner.fit_matched_bc = fit
    sys.argv = [sys.argv[0], *rest]
    try:
        conditioned.main()
    finally:
        runner.fit_matched_bc = original_fit
        original_fit.__globals__["optax"] = original_optax
        assert digest(args.bc_source) == source_hash


if __name__ == "__main__":
    main()
