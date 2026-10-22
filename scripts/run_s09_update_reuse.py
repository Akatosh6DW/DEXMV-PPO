"""Vary PPO minibatch reuse, not online interactions or network capacity."""

import argparse
import json
from pathlib import Path
import sys

import run_s09_constrained_ppo as constrained
from retarget_s09_reference import digest


def update_options(kwargs, passes):
    if passes not in (1, 2, 4):
        raise ValueError("Use one, two, or four passes")
    expected = dict(num_envs=32, num_minibatches=4, unroll_length=20,
                    batch_size=32, num_updates_per_batch=4)
    if any(kwargs.get(key) != value for key, value in expected.items()):
        raise ValueError("Keep the original 2560-transition, 20-step PPO batch")
    return dict(kwargs, num_updates_per_batch=passes)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--update-passes", type=int, choices=(1, 2, 4), required=True)
    args, rest = parser.parse_known_args()
    if any(x in rest for x in ("--teacher-kl", "--ppo-initial-std", "--ppo-discount")):
        parser.error("Do not combine sample reuse with additional interventions")
    runner = constrained.improvement.runner
    old_train = runner.train.train
    output = Path(rest[rest.index("--output")+1])

    def train(*values, **kwargs):
        options = update_options(kwargs, args.update_passes)
        metadata = dict(num_updates_per_batch=args.update_passes,
            minibatches_per_pass=4, gradient_updates_per_batch=4*args.update_passes,
            transitions_per_batch=2560, transitions_per_minibatch=640,
            unroll_length=20, online_budget_unchanged=True,
            gradient_budget_matches_original=args.update_passes == 4,
            source_sha256=digest(output/"sources"/Path(__file__).name))
        (output/"update_reuse_config.json").write_text(json.dumps(metadata, indent=2)+"\n")
        return old_train(*values, **options)

    runner.train.train = train
    sys.argv = [sys.argv[0], *rest]
    try:
        constrained.main()
    finally:
        runner.train.train = old_train


if __name__ == "__main__":
    main()
