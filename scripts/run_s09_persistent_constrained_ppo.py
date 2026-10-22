"""Test the existing R2 hold reward with the split actor/critic optimizer.

Keep this composition separate from the frozen R0 confirmation. Neither the
existing reward implementation nor any shared training source is edited.
"""

import argparse
import functools
import json
from pathlib import Path
import sys

import run_s09_constrained_ppo as constrained
import run_s09_persistent_reward as persistent
from retarget_s09_reference import digest


def record_reward(output, weight):
    source = output / "sources" / Path(persistent.__file__).name
    composition = output / "sources" / Path(__file__).name
    metadata = dict(reward_version="S09-R2_qualified_hold_progress",
        hold_progress_weight=weight,
        reward_formula="R0 + weight * dt * min(continuous_valid_steps/600,1) * valid * clean_history",
        reward_source_sha256=digest(source if source.exists() else Path(persistent.__file__)),
        composition_source_sha256=digest(composition if composition.exists() else Path(__file__)))
    for name in ("constraint_config.json", "improvement_config.json"):
        path = output / name
        if path.exists():
            values = json.loads(path.read_text())
            path.write_text(json.dumps(dict(values, **metadata), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hold-bonus-weight", type=float, default=15.)
    args, rest = parser.parse_known_args()
    if not 0 < args.hold_bonus_weight <= 100:
        parser.error("Hold bonus weight must lie in (0,100]")
    if "--method" not in rest or rest[rest.index("--method") + 1] != "guided":
        parser.error("This reward/optimizer probe uses full-target BC+PPO")
    if "--teacher-kl" in rest or "--ppo-discount" in rest:
        parser.error("Do not add teacher or discount interventions to this probe")
    output = Path(rest[rest.index("--output") + 1])
    improvement = constrained.improvement
    runner = improvement.runner
    original_train = runner.train.train
    original_conditioned, original_env = improvement.ModeConditionedGrasp, runner.OrcaS09Grasp

    def train(*values, **kwargs):
        # The split optimizer has written its metadata by this point.
        record_reward(output, args.hold_bonus_weight)
        return original_train(*values, **kwargs)

    runner.train.train = train
    improvement.ModeConditionedGrasp = functools.partial(
        persistent.PersistentConditionedGrasp, hold_weight=args.hold_bonus_weight)
    sys.argv = [sys.argv[0], *rest]
    try:
        constrained.main()
    finally:
        runner.train.train = original_train
        improvement.ModeConditionedGrasp, runner.OrcaS09Grasp = original_conditioned, original_env
        record_reward(output, args.hold_bonus_weight)


if __name__ == "__main__":
    main()
