"""Isolated BC+PPO repair experiments; preserve the frozen S09 implementation."""

import argparse
import functools
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import losses

import run_s09_job as runner
from retarget_s09_reference import digest
from orca_s09_bc_observation import RepairObservationGrasp


def anchored_loss(original, arrays, weight):
    """Keep acquisition and holding equally represented in the PPO BC anchor."""
    early_obs, early_action, late_obs, late_action = arrays[:4]

    def loss(params, normalizer_params, data, rng, *, ppo_network, **kwargs):
        value, metrics = original(params, normalizer_params, data, rng,
                                  ppo_network=ppo_network, **kwargs)
        ka, kb = jax.random.split(jax.random.fold_in(rng, 8101))
        i = jax.random.randint(ka, (128,), 0, len(early_obs))
        j = jax.random.randint(kb, (128,), 0, len(late_obs))
        obs = jp.concatenate([early_obs[i], late_obs[j]])
        target = jp.arctanh(jp.clip(jp.concatenate([early_action[i], late_action[j]]), -.995, .995))
        logits = ppo_network.policy_network.apply(normalizer_params, params.policy, {"state": obs})
        imitation = jp.mean((logits[..., :17] - target) ** 2)
        total = value + weight * imitation
        return total, dict(metrics, ppo_loss=value, demo_loss=imitation,
                           demo_weighted_loss=weight * imitation, total_loss=total)
    return loss


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repair-demo-weight", type=float, default=0.)
    parser.add_argument("--repair-acquisition-steps", type=int, default=30)
    parser.add_argument("--repair-observation", choices=("original", "no_history", "initial_axis"), default="original")
    parser.add_argument("--output", type=Path, required=True)
    args, rest = parser.parse_known_args()
    root = runner.ROOT / "artifacts/orca_benchmark_suite/S09/bc_repair"
    if not args.output.resolve().is_relative_to(root):
        parser.error("Repair runs must be isolated under S09/bc_repair")
    if args.repair_demo_weight < 0:
        parser.error("Demonstration weight must be nonnegative")
    if not 1 <= args.repair_acquisition_steps < 800:
        parser.error("Acquisition split must lie within the 800-step episode")
    if "--method" not in rest or rest[rest.index("--method") + 1] != "guided":
        parser.error("This entry point is only for BC+PPO, not residual control")
    original_arrays, original_loss = runner.dataset_arrays, losses.compute_ppo_loss
    loaded = []

    def arrays(*values, **kwargs):
        result = original_arrays(*values, **kwargs)
        manifest = json.loads((Path(values[0]).parent / "dataset_manifest.json").read_text())
        if args.repair_acquisition_steps != 30 or "train_episodes" in manifest or args.repair_observation != "original":
            with np.load(values[0], allow_pickle=False) as z:
                obs, action, active = z["obs"], z["action"], z["active"].astype(bool)
            if args.repair_observation == "no_history":
                obs[..., 34:68] = 0
            elif args.repair_observation == "initial_axis":
                obs[..., 95:98] = obs[:, :1, 74:77]
            split = manifest.get("train_episodes", len(obs) * 3 // 4)
            if not 0 < split < len(obs):
                raise ValueError("Invalid explicit train/validation split")
            early = np.arange(obs.shape[1])[None, :] < args.repair_acquisition_steps
            training = np.arange(len(obs))[:, None] < split
            masks = (active & training & early, active & training & ~early)
            if not all(mask.any() for mask in masks):
                raise ValueError("Missing early or late training samples")
            masks = (*masks, active & ~training)
            result = tuple(jp.asarray(x) for mask in masks for x in (obs[mask], action[mask]))
        loaded.append(result)
        if args.repair_demo_weight:
            losses.compute_ppo_loss = anchored_loss(original_loss, result, args.repair_demo_weight)
        return result

    runner.dataset_arrays = arrays
    physics_backend = runner.OrcaS09Grasp.physics_backend
    if args.repair_observation != "original":
        runner.OrcaS09Grasp = functools.partial(RepairObservationGrasp, observation_mode=args.repair_observation)
    # A resumed run bypasses BC; load the same validated train-only anchor explicitly.
    if args.repair_demo_weight and "--checkpoint" in rest:
        inp = Path(rest[rest.index("--input") + 1])
        dataset = Path(rest[rest.index("--dataset") + 1])
        arrays(dataset, True, digest(inp / "model.xml"), digest(inp / "reference.npz"),
               physics_backend)
    sys.argv = [sys.argv[0], *rest, "--output", str(args.output)]
    try:
        runner.main()
    finally:
        losses.compute_ppo_loss = original_loss
        if args.output.exists():
            snapshot = args.output / "sources" / Path(__file__).name
            metadata = dict(demo_weight=args.repair_demo_weight,
                source_sha256=digest(snapshot if snapshot.exists() else Path(__file__)),
                acquisition_steps=args.repair_acquisition_steps,
                observation_mode=args.repair_observation,
                balanced_early_late=True, anchor_loaded=bool(loaded),
                action_architecture="Full absolute targets from policy; NO reference-plus-residual execution")
            (args.output / "repair_config.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
