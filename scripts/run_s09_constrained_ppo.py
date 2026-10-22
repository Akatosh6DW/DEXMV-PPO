"""Separate critic speed or anchor the actor during S09 PPO fine-tuning.

The optional teacher participates only in the loss. Evaluation uses the same
full-target or residual action interface as the corresponding frozen baseline.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import jax
import optax
from brax.training.agents.ppo import losses

import run_s09_ppo_improvement as improvement
from run_s09_bc_trust import trusted_loss
from retarget_s09_reference import digest


def split_adam(actor_rate, critic_rate):
    def labels(params):
        return params.replace(
            policy=jax.tree_util.tree_map(lambda _: "actor", params.policy),
            value=jax.tree_util.tree_map(lambda _: "critic", params.value))
    return optax.multi_transform(
        dict(actor=optax.adam(actor_rate), critic=optax.adam(critic_rate)), labels)


class OptimizerProxy:
    """Override Brax's optimizer factory locally, not the shared optax module."""

    def __init__(self, original, critic_rate):
        self.original, self.critic_rate = original, critic_rate

    def __getattr__(self, name):
        return getattr(self.original, name)

    def adam(self, learning_rate):
        return split_adam(learning_rate, self.critic_rate)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--critic-lr", type=float)
    parser.add_argument("--teacher-kl", type=float, default=0.)
    args, rest = parser.parse_known_args()
    if not math.isfinite(args.teacher_kl) or args.teacher_kl < 0:
        parser.error("Teacher KL weight must be finite and nonnegative")
    if args.critic_lr is not None and not 0 < args.critic_lr < 1:
        parser.error("Critic learning rate must be between zero and one")
    runner = improvement.runner
    original_train, original_loss = runner.train.train, losses.compute_ppo_loss
    original_optax = runner.train.optax
    output = Path(rest[rest.index("--output")+1])
    metadata = dict(critic_learning_rate=args.critic_lr, teacher_kl_weight=args.teacher_kl,
                    teacher_used_at_inference=False, reward_version="S09-R0 unchanged",
                    training_episode_logging=True)

    def train(*values, **kwargs):
        kwargs.update(log_training_metrics=True, training_metrics_steps=10240)
        # The outer improvement wrapper has already applied any std override.
        if args.teacher_kl:
            losses.compute_ppo_loss = trusted_loss(original_loss, kwargs["restore_params"], args.teacher_kl)
        if args.critic_lr is not None:
            if kwargs.get("learning_rate_schedule"):
                raise ValueError("Do not combine split Adam with adaptive global LR")
            runner.train.optax = OptimizerProxy(original_optax, args.critic_lr)
        metadata["actor_learning_rate"] = kwargs["learning_rate"]
        snapshot = output/"sources"/Path(__file__).name
        metadata["source_sha256"] = digest(snapshot)
        (output/"constraint_config.json").write_text(json.dumps(metadata, indent=2)+"\n")
        return original_train(*values, **kwargs)

    runner.train.train = train
    sys.argv = [sys.argv[0], *rest]
    try:
        improvement.main()
    finally:
        runner.train.train = original_train
        runner.train.optax = original_optax
        losses.compute_ppo_loss = original_loss


if __name__ == "__main__":
    main()
