"""Explore more before release, without adding noise to deterministic deployment.

This is a separate follow-up probe. The frozen confirmation and the earlier
BC-only noise-reduction experiment remain unchanged.
"""

import argparse
import dataclasses
import json
import math
from pathlib import Path
import sys

import jax
import jax.numpy as jp
from brax.training.distribution import NormalTanhDistribution

import run_s09_temporal_ppo as temporal
from retarget_s09_reference import digest


def acquisition_logits(logits, observation, multiplier):
    if not math.isfinite(multiplier) or not 1 <= multiplier <= 10:
        raise ValueError("Acquisition multiplier must lie in [1,10]")
    if logits.shape[-1] != 34 or observation.shape[-1] != 123:
        raise ValueError("Expected the S09 17-action/123-input interface")
    if multiplier == 1:
        return logits
    mean, raw_std = jp.split(logits, 2, axis=-1)
    desired = (jax.nn.softplus(raw_std) + .001) * multiplier - .001
    # Invert softplus stably; use the same logits in sampling and PPO loss.
    early_raw = desired + jp.log(-jp.expm1(-desired))
    std_logits = jp.where(observation[..., 93:94] < 1, early_raw, raw_std)
    return jp.concatenate([mean, std_logits], axis=-1)


def wrap_network(network, multiplier):
    distribution = network.parametric_action_distribution
    if (not isinstance(distribution, NormalTanhDistribution)
            or distribution._min_std != .001 or distribution._var_scale != 1):
        raise ValueError("Unexpected action-distribution parameterization")
    original = network.policy_network

    def apply(normalizer, params, observation):
        logits = original.apply(normalizer, params, observation)
        return acquisition_logits(logits, observation["state"], multiplier)

    return dataclasses.replace(network,
        policy_network=dataclasses.replace(original, apply=apply))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--acquisition-noise-multiplier", type=float, default=10.)
    parser.add_argument("--rollout-length", type=int, default=80, choices=(80,))
    args, rest = parser.parse_known_args()
    multiplier = args.acquisition_noise_multiplier
    if not math.isfinite(multiplier) or not 1 <= multiplier <= 10:
        parser.error("Acquisition multiplier must lie in [1,10]")
    if "--method" not in rest or rest[rest.index("--method") + 1] != "residual":
        parser.error("This probe is residual PPO only")
    if "--teacher-kl" in rest:
        parser.error("Keep teacher regularization out of this isolated probe")
    output = Path(rest[rest.index("--output") + 1])
    runner = temporal.constrained.improvement.runner
    original_train = runner.train.train

    def train(*values, **kwargs):
        if kwargs["normalize_observations"]:
            raise ValueError("The phase gate requires raw, unnormalized observations")
        original_factory = kwargs["network_factory"]

        def factory(*args_, **kwargs_):
            return wrap_network(original_factory(*args_, **kwargs_), multiplier)

        kwargs["network_factory"] = factory
        metadata = dict(acquisition_noise_scale=multiplier, phase_input_index=93,
            early_phase_upper_exclusive=1., acquisition_seconds=3.,
            sampling_and_loss_same_distribution=True, deterministic_mean_unchanged=True,
            deployment_uses_wrapper=False, method="residual", residual_bound_rad=.08,
            difference_from_control="For fixed parameters and observations, only pre-release sampling std changes and late logits remain bit-identical. Subsequent learning can change all policy means.",
            source_sha256=digest(output / "sources" / Path(__file__).name))
        (output / "exploration_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return original_train(*values, **kwargs)

    runner.train.train = train
    sys.argv = [sys.argv[0], *rest, "--rollout-length", str(args.rollout_length)]
    try:
        temporal.main()
    finally:
        runner.train.train = original_train


if __name__ == "__main__":
    main()
