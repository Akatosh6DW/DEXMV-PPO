"""PPO with optional low exploration and on-policy KL to its BC initialization.

The teacher is used in the TRAINING LOSS only. Deployment remains one learned
full-target network, without a teacher action addition or reference replay.
"""

import argparse
import copy
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import losses

import run_s09_bc_conditioned as conditioned
import run_s09_bc_repair as repair
from retarget_s09_reference import digest


def with_std(params, std):
    if std is None:
        return params
    if std <= .001:
        raise ValueError("NormalTanh minimum std is .001")
    updated = copy.deepcopy(params)
    head = updated[1]["params"]["hidden_2"]
    head["kernel"] = jp.asarray(head["kernel"]).at[:,17:].set(0)
    head["bias"] = jp.asarray(head["bias"]).at[17:].set(np.log(np.expm1(std-.001)))
    return updated


def trusted_loss(original, teacher, weight):
    def loss(params, normalizer_params, data, rng, *, ppo_network, **kwargs):
        value, metrics = original(params, normalizer_params, data, rng, ppo_network=ppo_network, **kwargs)
        logits = ppo_network.policy_network.apply(normalizer_params, params.policy, data.observation)
        old_logits = jax.lax.stop_gradient(ppo_network.policy_network.apply(teacher[0], teacher[1], data.observation))
        distribution = ppo_network.parametric_action_distribution
        current, frozen = distribution.create_dist(logits), distribution.create_dist(old_logits)
        # Both scales are >= .001. No epsilon is needed inside log(ratio), which
        # would make KL(P||P) nonzero. Match Brax's KL(teacher || current) direction.
        ratio = frozen.scale / current.scale
        standardized = (frozen.loc-current.loc) / current.scale
        kl = jp.mean(jp.sum(.5*(ratio**2+standardized**2-1)-jp.log(ratio), axis=-1))
        total = value + weight * kl
        return total, dict(metrics, bc_teacher_kl=kl, bc_teacher_penalty=weight*kl, total_loss=total)
    return loss


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trust-std", type=float)
    parser.add_argument("--trust-weight", type=float, default=0.)
    parser.add_argument("--conditioned", action="store_true")
    args, rest = parser.parse_known_args()
    if args.trust_weight < 0 or "--repair-demo-weight" in rest:
        parser.error("Use a nonnegative KL weight; do not confound with an offline demo anchor")
    output = Path(rest[rest.index("--output")+1])
    original_train, original_loss = repair.runner.train.train, losses.compute_ppo_loss
    metadata = dict(initial_std_override=args.trust_std, teacher_kl_weight=args.trust_weight,
        teacher_used_at_inference=False, conditioned=args.conditioned,
        action_architecture="single network, full absolute joint targets")

    def train(*values, **kwargs):
        params = with_std(kwargs["restore_params"], args.trust_std)
        kwargs["restore_params"] = params
        if args.trust_weight:
            losses.compute_ppo_loss = trusted_loss(original_loss, params, args.trust_weight)
        return original_train(*values, **kwargs)

    repair.runner.train.train = train
    sys.argv = [sys.argv[0], *rest]
    try:
        (conditioned.main if args.conditioned else repair.main)()
    finally:
        losses.compute_ppo_loss = original_loss
        repair.runner.train.train = original_train
        if output.exists():
            snapshot = output/"sources"/Path(__file__).name
            metadata["source_sha256"] = digest(snapshot if snapshot.exists() else Path(__file__))
            (output/"trust_config.json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
