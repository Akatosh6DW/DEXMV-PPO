"""Reduce acquisition exploration without changing the deterministic policy.

The same transformed distribution is used for sampling and PPO likelihoods.
Only training uses this wrapper; deployment still predicts full joint targets.
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

import run_s09_constrained_ppo as constrained
from retarget_s09_reference import digest


def phase_logits(logits, observation, acquisition_scale):
    if not math.isfinite(acquisition_scale) or not 0 < acquisition_scale <= 1:
        raise ValueError("Acquisition noise multiplier must lie in (0,1]")
    if logits.shape[-1] != 34 or observation.shape[-1] != 123:
        raise ValueError("This experiment requires the S09 17-action/123-input interface")
    mean, raw_std = jp.split(logits, 2, axis=-1)
    std = jax.nn.softplus(raw_std) + .001
    desired = jp.maximum(std * acquisition_scale - .001, 1e-6)
    # Stable inverse softplus, with the distribution's existing minimum retained.
    early_raw_std = desired + jp.log(-jp.expm1(-desired))
    transformed = jp.where(observation[..., 93:94] < 1, early_raw_std, raw_std)
    return jp.concatenate([mean, transformed], axis=-1)


def wrap_network(network, acquisition_scale):
    distribution = network.parametric_action_distribution
    if (not isinstance(distribution, NormalTanhDistribution)
            or distribution._min_std != .001 or distribution._var_scale != 1):
        raise ValueError("Unexpected action distribution; do not silently rescale it")
    original = network.policy_network

    def apply(normalizer, params, observation):
        logits = original.apply(normalizer, params, observation)
        return phase_logits(logits, observation["state"], acquisition_scale)

    policy = dataclasses.replace(original, apply=apply)
    return dataclasses.replace(network, policy_network=policy)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--acquisition-noise-scale", type=float, default=.1)
    args, rest = parser.parse_known_args()
    if not math.isfinite(args.acquisition_noise_scale) or not 0 < args.acquisition_noise_scale <= 1:
        parser.error("Acquisition noise multiplier must lie in (0,1]")
    if rest[rest.index("--method") + 1] != "guided" or "--teacher-kl" in rest:
        parser.error("This isolated probe requires full-target BC+PPO without teacher KL")
    output = Path(rest[rest.index("--output") + 1])
    runner = constrained.improvement.runner
    original_train = runner.train.train

    def train(*values, **kwargs):
        original_factory = kwargs["network_factory"]

        def factory(*args_, **kwargs_):
            return wrap_network(original_factory(*args_, **kwargs_), args.acquisition_noise_scale)

        kwargs["network_factory"] = factory
        metadata = dict(acquisition_noise_scale=args.acquisition_noise_scale,
            phase_input_index=93, early_phase_upper_exclusive=1., acquisition_seconds=3.,
            minimum_effective_std=.001001, sampling_and_loss_same_distribution=True,
            deterministic_mean_unchanged=True, deployment_uses_wrapper=False,
            source_sha256=digest(output / "sources" / Path(__file__).name))
        (output / "exploration_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return original_train(*values, **kwargs)

    runner.train.train = train
    sys.argv = [sys.argv[0], *rest]
    try:
        constrained.main()
    finally:
        runner.train.train = original_train


if __name__ == "__main__":
    main()
