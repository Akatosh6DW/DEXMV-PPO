"""Change training reset position jitter, leaving the evaluation env untouched."""

import argparse
import json
import math
from pathlib import Path
import sys

import run_s09_temporal_ppo as temporal
from retarget_s09_reference import digest


def training_environment(evaluation, jitter):
    if not math.isfinite(jitter) or not .002 <= jitter <= .01:
        raise ValueError("Training jitter must lie in [0.002,0.01] meters")
    if evaluation.jitter != .002 or evaluation.orientation_mode != "uniform":
        raise ValueError("Preserve the original uniform-orientation, 2mm evaluation")
    reference = evaluation.path.parent/"reference.npz" if evaluation.arm == "residual" else None
    return type(evaluation)(evaluation.path, arm=evaluation.arm, reference=reference,
        jitter=jitter, residual_bound=evaluation.residual_bound,
        approach_weight=evaluation.approach_weight, coverage_weight=evaluation.coverage_weight,
        orientation_mode=evaluation.orientation_mode)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--training-jitter", type=float, default=.005)
    args, rest = parser.parse_known_args()
    if not math.isfinite(args.training_jitter) or not .002 <= args.training_jitter <= .01:
        parser.error("Training jitter must lie in [0.002,0.01] meters")
    if "--rollout-length" not in rest or "--teacher-kl" in rest or "--ppo-discount" in rest:
        parser.error("Specify the matched rollout length; keep teacher and gamma unchanged")
    output = Path(rest[rest.index("--output")+1])
    runner = temporal.constrained.improvement.runner
    old_train = runner.train.train

    def train(*values, **kwargs):
        evaluation = kwargs["environment"]
        training = training_environment(evaluation, args.training_jitter)
        assert training is not evaluation and training.native is not evaluation.native
        kwargs["environment"] = training
        metadata = dict(training_env_config=training._config.to_dict(),
            evaluation_env_config=evaluation._config.to_dict(),
            training_jitter_m=training.jitter, evaluation_jitter_m=evaluation.jitter,
            separate_native_physics_objects=True, evaluation_callback_environment_unchanged=True,
            reward_version="S09-R0 unchanged", source_sha256=digest(output/"sources"/Path(__file__).name))
        (output/"training_jitter_config.json").write_text(json.dumps(metadata, indent=2)+"\n")
        return old_train(*values, **kwargs)

    runner.train.train = train
    sys.argv = [sys.argv[0], *rest]
    try:
        temporal.main()
    finally:
        runner.train.train = old_train


if __name__ == "__main__":
    main()
