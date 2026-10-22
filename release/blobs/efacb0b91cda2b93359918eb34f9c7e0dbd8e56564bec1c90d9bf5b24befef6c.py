"""Isolated PPO optimization experiments on the unchanged S09 grasp task.

Guided policies keep the repaired full-target observation interface. Residual
policies keep reference + bounded correction. Neither action law is changed.
"""

import argparse
import json
from pathlib import Path
import sys

import run_s09_job as runner
from run_s09_bc_conditioned import ModeConditionedGrasp
from run_s09_bc_trust import with_std
from retarget_s09_reference import digest


def training_options(discount):
    if not 0 < discount < 1:
        raise ValueError("Discount must be strictly between zero and one")
    return dict(discounting=float(discount))


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--ppo-discount", type=float, default=.99)
    p.add_argument("--ppo-initial-std", type=float)
    args, rest = p.parse_known_args()
    options = training_options(args.ppo_discount)
    if args.ppo_initial_std is not None and not .001 < args.ppo_initial_std <= 1:
        p.error("Exploration scale must lie in (.001,1]")
    output = Path(rest[rest.index("--output")+1])
    root = runner.ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"
    if not output.resolve().is_relative_to(root):
        p.error("Keep this experiment separate from all frozen S09 reports")
    method = rest[rest.index("--method")+1]
    if method not in ("guided", "residual", "reference"):
        p.error("Use repaired BC+PPO, residual PPO, or its reference baseline")
    if method == "guided":
        if "--checkpoint" not in rest:
            p.error("Use an explicit, already verified repaired BC checkpoint")
        checkpoint = Path(rest[rest.index("--checkpoint")+1])
        parent = checkpoint.parent
        if not (parent/"run_config.json").exists():
            parent = parent.parent
        previous = json.loads((parent/"run_config.json").read_text())
        if previous["env_config"].get("observation_variant") != "initial_pose_mode_onehot_95_101":
            p.error("Repaired BC requires the remembered initial-pose observation interface")
        runner.OrcaS09Grasp = ModeConditionedGrasp
    original_train = runner.train.train

    def train(*values, **kwargs):
        # Override only the explicitly tested optimizer parameter. The frozen
        # runner continues to own rollout length, rewards, and evaluation.
        kwargs.update(options)
        if args.ppo_initial_std is not None:
            kwargs["restore_params"] = with_std(kwargs["restore_params"], args.ppo_initial_std)
        return original_train(*values, **kwargs)

    runner.train.train = train
    sys.argv = [sys.argv[0], *rest]
    try:
        runner.main()
    finally:
        runner.train.train = original_train
        if output.exists():
            snapshot = output/"sources"/Path(__file__).name
            metadata = dict(training_overrides=options, reward_version="S09-R0 unchanged",
                initial_std_override=args.ppo_initial_std,
                conditioned=method=="guided", action_architecture=(
                    "single network full joint targets" if method=="guided" else "reference plus bounded residual"),
                source_sha256=digest(snapshot if snapshot.exists() else Path(__file__)))
            (output/"improvement_config.json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
