"""Anchor reliable BC orientation modes during full-target PPO fine-tuning.

The teacher appears only in the training loss. Deployment is the unchanged
single actor, not a reference action plus a residual actor.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import jax
import jax.numpy as jp
from brax.training.agents.ppo import losses

import run_s09_constrained_ppo as constrained
from retarget_s09_reference import digest


def mode_trusted_loss(original, teacher, weight, modes):
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("Use a finite positive anchor weight")
    if not modes or len(set(modes)) != len(modes) or any(m not in range(6) for m in modes):
        raise ValueError("Select distinct mode indices in 0..5")
    selected = jp.asarray([i in modes for i in range(6)])

    def loss(params, normalizer_params, data, rng, *, ppo_network, **kwargs):
        value, metrics = original(params, normalizer_params, data, rng,
                                 ppo_network=ppo_network, **kwargs)
        obs = data.observation["state"]
        if obs.shape[-1] != 123:
            raise ValueError("Expected raw repaired BC observations")
        logits = ppo_network.policy_network.apply(normalizer_params, params.policy, data.observation)
        old = jax.lax.stop_gradient(ppo_network.policy_network.apply(
            teacher[0], teacher[1], data.observation))
        dist = ppo_network.parametric_action_distribution
        current, frozen = dist.create_dist(logits), dist.create_dist(old)
        ratio = frozen.scale / current.scale
        deviation = (frozen.loc-current.loc) / current.scale
        per_state = jp.sum(.5*(ratio**2+deviation**2-1)-jp.log(ratio), axis=-1)
        gate = jp.sum(obs[..., 95:101] * selected, axis=-1)
        # Average over the full batch, as with the ungated teacher loss.
        # No selected samples means zero penalty, not division by zero.
        kl = jp.mean(per_state * gate)
        total = value + weight * kl
        return total, dict(metrics, bc_teacher_kl=kl, bc_teacher_penalty=weight*kl,
                           anchor_fraction=jp.mean(gate), total_loss=total)
    return loss


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--anchor-modes", type=int, nargs="+", default=[0])
    parser.add_argument("--anchor-weight", type=float, default=.1)
    args, rest = parser.parse_known_args()
    mode_trusted_loss(None, None, args.anchor_weight, args.anchor_modes)
    if "--method" not in rest or rest[rest.index("--method")+1] != "guided":
        parser.error("This experiment uses a full-target BC-initialized actor")
    if "--critic-lr" not in rest or "--teacher-kl" in rest:
        parser.error("Use split rates without a second teacher penalty")
    output = Path(rest[rest.index("--output")+1])
    runner = constrained.improvement.runner
    old_train, old_loss = runner.train.train, losses.compute_ppo_loss

    def train(*values, **kwargs):
        if kwargs.get("normalize_observations"):
            raise ValueError("Mode gate requires raw one-hot observations")
        teacher = kwargs["restore_params"]
        losses.compute_ppo_loss = mode_trusted_loss(
            old_loss, teacher, args.anchor_weight, args.anchor_modes)
        metadata = dict(anchor_modes=args.anchor_modes, anchor_weight=args.anchor_weight,
            teacher="restored stage-start actor", gate="raw initial mode onehot columns 95:101",
            reduction="mean across all rollout samples, unselected samples contribute zero",
            teacher_used_at_inference=False, reference_action_added=False,
            reward_version="S09-R0 unchanged", source_sha256=digest(output/"sources"/Path(__file__).name))
        (output/"mode_anchor_config.json").write_text(json.dumps(metadata, indent=2)+"\n")
        return old_train(*values, **kwargs)

    runner.train.train = train
    sys.argv = [sys.argv[0], *rest]
    try:
        constrained.main()
    finally:
        runner.train.train, losses.compute_ppo_loss = old_train, old_loss


if __name__ == "__main__":
    main()
