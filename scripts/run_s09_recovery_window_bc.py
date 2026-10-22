"""Oversample verified recovery onset, leaving data and BC budget unchanged."""

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jp
import numpy as np
import optax

import run_s09_recovery_bc as recovery
from retarget_s09_reference import digest


def window_indices(active, original_split, split, start=40, end=80):
    active = np.asarray(active, bool)
    if active.ndim != 2 or not 0 < original_split < split < len(active):
        raise ValueError("Expected new training recoveries before unchanged validation")
    if not 30 <= start < end <= active.shape[1]:
        raise ValueError("Recovery window must be within late training frames")
    late = active & (np.arange(len(active))[:, None] < split) & (np.arange(active.shape[1])[None, :] >= 30)
    window = np.zeros_like(late)
    window[original_split:split, start:end] = True
    indices = np.flatnonzero(window[late])
    if not len(indices):
        raise ValueError("No active recovery onset samples")
    return indices, int(late.sum())


def sample_late(key, size, eligible, count):
    indices = jax.random.randint(key, (128,), 0, size)
    if count:
        selected = jax.random.randint(jax.random.fold_in(key, 918), (count,), 0, len(eligible))
        indices = indices.at[:count].set(eligible[selected])
    return indices


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--recovery-samples", type=int, default=16)
    args, rest = parser.parse_known_args()
    if not 0 <= args.recovery_samples <= 128:
        parser.error("Recovery samples must be within the unchanged 128 late samples")
    dataset = Path(rest[rest.index("--dataset")+1])
    output = Path(rest[rest.index("--output")+1])
    summary = json.loads((dataset.parent/"summary.json").read_text())
    assert summary["status"] == "completed" and summary["original_validation_preserved"]
    assert summary["dataset_sha256"] == digest(dataset)
    with np.load(dataset, allow_pickle=False) as z:
        indices, late_count = window_indices(z["active"], summary["original_train_episodes"], summary["new_train_episodes"])
    runner = recovery.conditioned.repair.runner
    original = runner.fit_matched_bc

    def fit(net, params, arrays, updates, seed):
        early_obs, early_action, late_obs, late_action, val_obs, val_action = arrays
        assert len(late_obs) == late_count
        # recovery.main supplies the learning-rate proxy and original I27 params.
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params[1])

        def loss(policy, obs, actions):
            logits = net.policy_network.apply(params[0], policy, {"state": obs})
            return jp.mean((logits[..., :17]-jp.arctanh(jp.clip(actions, -.995, .995)))**2)

        @jax.jit
        def update(policy, opt_state, key):
            a, b = jax.random.split(key)
            i = jax.random.randint(a, (128,), 0, len(early_obs))
            j = sample_late(b, len(late_obs), jp.asarray(indices), args.recovery_samples)
            value, grads = jax.value_and_grad(loss)(policy, jp.concatenate([early_obs[i], late_obs[j]]),
                                                    jp.concatenate([early_action[i], late_action[j]]))
            change, opt_state = optimizer.update(grads, opt_state, policy)
            return optax.apply_updates(policy, change), opt_state, value

        validation = jax.jit(lambda policy: loss(policy, val_obs, val_action))
        policy, key, rows = params[1], jax.random.PRNGKey(seed+5000), []
        for step in range(updates):
            key, subkey = jax.random.split(key)
            policy, opt_state, value = update(policy, opt_state, subkey)
            if step % 100 == 0 or step == updates-1:
                rows.append(dict(step=step+1, train_loss=float(value), validation_loss=float(validation(policy))))
        metadata = dict(recovery_samples_per_batch=args.recovery_samples, total_batch=256,
            early_samples=128, late_samples=128, recovery_window_steps=[40, 80],
            recovery_window_frames=len(indices), late_frames=late_count,
            old_expected_recovery_samples=128*len(indices)/late_count,
            new_expected_recovery_samples=args.recovery_samples+(128-args.recovery_samples)*len(indices)/late_count,
            dataset_sha256=digest(dataset), policy_updates=updates, ppo_steps=0,
            source_sha256=digest(Path(__file__)),
            limitation="Custom supervised sampling ablation; not full DAgger, DART or PPO")
        (output/"recovery_window_config.json").write_text(json.dumps(metadata, indent=2)+"\n")
        return (params[0], policy, params[2]), rows

    runner.fit_matched_bc = fit
    sys.argv = [sys.argv[0], *rest]
    try:
        recovery.main()
    finally:
        runner.fit_matched_bc = original


if __name__ == "__main__":
    main()
