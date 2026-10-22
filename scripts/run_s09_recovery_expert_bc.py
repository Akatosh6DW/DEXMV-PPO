"""Fit six pose-routed BC actors on the unchanged recovery data/minibatches.

No PPO updates and no reference action is added at deployment. The actors have
six times the parameter capacity; this is not a capacity-matched ablation.
"""

import json
from pathlib import Path
import sys

import jax
import numpy as np
import optax

import run_s09_recovery_bc as recovery
import run_s09_mode_experts as experts
from make_s09_mode_experts import verify_copies
from retarget_s09_reference import digest


def fit_expanded(original, net, params, arrays, updates, seed, optimizer_module):
    expanded = experts.expand_parameters(params)
    verify_copies(params, expanded)
    old = original.__globals__["optax"]
    original.__globals__["optax"] = optimizer_module
    try:
        return original(net, expanded, arrays, updates, seed)
    finally:
        original.__globals__["optax"] = old


def main():
    if any(flag in sys.argv for flag in ("--phase-only", "--feedback-source", "--checkpoint")):
        raise ValueError("Use the unchanged phase-only I27 source and full-feedback BC")
    output = Path(sys.argv[sys.argv.index("--output")+1])
    runner = recovery.conditioned.repair.runner
    original_fit = runner.fit_matched_bc

    def fit(net, params, arrays, updates, seed):
        counts = sum(x.size for x in jax.tree_util.tree_leaves(params[1]))
        modes = [np.argmax(np.asarray(arrays[i])[:, 95:101], axis=1) for i in (0, 2, 4)]
        record = dict(policy_architecture=experts.ARCHITECTURE, routing="argmax(raw_state[95:101])",
            expert_count=6, total_actor_parameters=6*counts, original_actor_parameters=counts,
            initial_six_copies_identical=True, critic_and_normalizer_unchanged=True,
            training_sampling_unchanged=True, bc_updates=updates, ppo_steps=0,
            early_mode_counts=np.bincount(modes[0], minlength=6).tolist(),
            late_mode_counts=np.bincount(modes[1], minlength=6).tolist(),
            validation_mode_counts=np.bincount(modes[2], minlength=6).tolist(),
            source_sha256=digest(Path(__file__)),
            limitation="Sixfold actor capacity; BC specialization only, not full divide-and-conquer RL")
        (output/"recovery_expert_config.json").write_text(json.dumps(record, indent=2)+"\n")
        # recovery.main injects its requested BC rate into this module's optax.
        return fit_expanded(original_fit, net, params, arrays, updates, seed, optax)

    runner.fit_matched_bc = fit
    try:
        with experts.expert_networks():
            recovery.main()
    finally:
        runner.fit_matched_bc = original_fit
        path = output/"run_config.json"
        if path.exists():
            cfg = json.loads(path.read_text())
            cfg["policy_architecture"] = experts.ARCHITECTURE
            path.write_text(json.dumps(cfg, indent=2)+"\n")


if __name__ == "__main__":
    main()
