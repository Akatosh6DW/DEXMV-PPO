"""Measure every train-only demonstration frame on one fixed CPU backend.

This is an action-fit diagnostic, not a physical grasp-success evaluation.
Acquisition and holding receive equal weight, matching the rehearsal sampler.
"""

import argparse
import json
from pathlib import Path
import pickle

from run_s09_demo_rehearsal import load_rehearsal
from train_orca_comparison import network_factory
from retarget_s09_reference import digest
import jax
import jax.numpy as jp
import numpy as np


def mean_error(apply, normalizer, policy, obs, actions, batch_size=2048):
    total, count = 0., 0
    for start in range(0, len(obs), batch_size):
        logits = np.asarray(apply(normalizer, policy, obs[start:start+batch_size]))[..., :17]
        targets = np.arctanh(np.clip(np.asarray(actions[start:start+batch_size]), -.995, .995))
        if not np.isfinite(logits).all() or not np.isfinite(targets).all():
            raise ValueError("Nonfinite actor outputs or demonstration targets")
        total += np.square(logits-targets).sum(dtype=np.float64)
        count += targets.size
    return float(total/count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        parser.error("Use JAX_PLATFORMS=cpu for fixed-backend diagnostic comparability")
    jax.config.update("jax_default_matmul_precision", "highest")
    early_obs, early_action, late_obs, late_action = load_rehearsal(args.dataset, args.input)
    net = network_factory(dict(state=(123,), privileged_state=(140,)), 17)
    apply = jax.jit(lambda stats, policy, obs: net.policy_network.apply(stats, policy, {"state": obs}))
    rows = []
    for checkpoint in args.checkpoint:
        normalizer, policy, _ = pickle.loads(checkpoint.read_bytes())
        if "experts" in policy:
            parser.error("Use this diagnostic only for the original single actor")
        early = mean_error(apply, normalizer, policy, early_obs, early_action)
        late = mean_error(apply, normalizer, policy, late_obs, late_action)
        rows.append(dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest(checkpoint),
            acquisition_mean_logit_mse=early, holding_mean_logit_mse=late,
            balanced_mean_logit_mse=.5*(early+late)))
    report = dict(scope="Train-only demonstration action fitting, not physical success",
        backend="cpu", dataset_sha256=digest(args.dataset), early_frames=len(early_obs),
        holding_frames=len(late_obs), validation_used=False, target_action_clip=.995,
        source_sha256=digest(Path(__file__)), rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
