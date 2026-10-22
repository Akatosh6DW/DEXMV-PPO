"""Repeat the unchanged full-target audit using the evaluation batch layout.

Large flattened GEMMs can round differently from scan(vmap(policy)) over the
original eight-environment batches. Preserve seeds' original batch slots and
the scan layout instead of increasing the existing action tolerance.
"""

import argparse
import json
from pathlib import Path
import pickle

import audit_s09_bc_repair as audit
from retarget_s09_reference import digest
import jax
import jax.numpy as jp
import numpy as np


def batch_groups(seeds, start, count, batch_size):
    seeds = [int(seed) for seed in seeds]
    if len(set(seeds)) != len(seeds) or any(not start <= s < start+count for s in seeds):
        raise ValueError("Preview seeds must be unique members of the evaluation set")
    groups = {}
    for index, seed in enumerate(seeds):
        group, slot = divmod(seed-start, batch_size)
        groups.setdefault(group, []).append((index, slot))
    return list(groups.values())


def original_layout_actions(policy, observations, groups, batch_size):
    def episode(obs):
        def step(carry, value):
            action, _ = policy({"state": value}, jax.random.PRNGKey(0))
            return carry, action
        return jax.lax.scan(step, None, obs)[1]

    result = jp.zeros((*observations.shape[:2], 17), dtype=observations.dtype)
    for members in groups:
        padded = jp.zeros((batch_size, *observations.shape[1:]), observations.dtype)
        for index, slot in members:
            padded = padded.at[slot].set(observations[index])
        actions = jax.vmap(episode)(padded)
        for index, slot in members:
            result = result.at[index].set(actions[slot])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    cfg = json.loads((args.run/"run_config.json").read_text())
    if cfg["method"] != "guided" or cfg.get("policy_architecture"):
        parser.error("This checker is for the original single full-target actor")
    if jax.default_backend() != cfg["backend"]:
        parser.error("Use the recorded evaluation backend")
    jax.config.update("jax_default_matmul_precision", "highest")
    folder = args.run/f"step_{args.step:09d}"
    with np.load(folder/"preview_trajectories.npz", allow_pickle=False) as archive:
        trace = dict(archive)
    groups = batch_groups(trace["seeds"], cfg["eval_start"], cfg["eval_count"], cfg["eval_batch"])
    old_factory = audit.runner.networks.make_inference_fn
    errors = {}

    def factory(net, *values, **kwargs):
        make_policy = old_factory(net, *values, **kwargs)

        def make(params, deterministic=False):
            if not deterministic:
                raise ValueError("The audit must use deterministic deployment actions")
            policy = make_policy(params, deterministic=True)
            dense = jax.jit(lambda obs: policy({"state": obs}, jax.random.PRNGKey(0))[0])
            values = np.asarray(dense(jp.asarray(trace["obs"])))
            errors["dense_max_action_error"] = float(np.max(
                np.abs(values[trace["active"]]-trace["action"][trace["active"]])))

            def inference(obs, key):
                del key
                return original_layout_actions(policy, obs["state"], groups, cfg["eval_batch"]), {}

            return inference
        return make

    audit.runner.networks.make_inference_fn = factory
    try:
        audit.main()
    finally:
        audit.runner.networks.make_inference_fn = old_factory
    verified = json.loads((folder/"native_replay_audit.json").read_text())
    report = dict(errors, status=verified["status"], unchanged_action_tolerance=1e-5,
        original_eval_batch=cfg["eval_batch"], preview_groups=groups,
        layout_max_action_error=verified["max_policy_action_error"],
        original_trace_sha256=digest(folder/"preview_trajectories.npz"),
        checkpoint_sha256=digest(folder/"params.pkl"), source_sha256=digest(Path(__file__)),
        trace_and_actions_modified=False, physical_replay_tolerances_unchanged=True)
    (folder/"inference_layout_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
