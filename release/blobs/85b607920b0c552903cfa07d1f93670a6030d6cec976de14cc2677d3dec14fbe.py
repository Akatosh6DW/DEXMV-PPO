"""Change temporal partitioning without changing transitions per PPO batch.

Longer sequences reduce short-fragment bootstrapping. Fewer sequences keep the
same 2560 transitions per update batch and 640 transitions per minibatch.
"""

import argparse
import json
from pathlib import Path
import sys

import run_s09_constrained_ppo as constrained
from retarget_s09_reference import digest


def temporal_options(length):
    if length not in (20, 40, 80):
        raise ValueError("Use a verified temporal partition: 20, 40, or 80")
    return dict(unroll_length=length, batch_size=640 // length)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rollout-length", type=int, default=80, choices=(20, 40, 80))
    args, rest = parser.parse_known_args()
    options = temporal_options(args.rollout_length)
    output = Path(rest[rest.index("--output") + 1])
    runner = constrained.improvement.runner
    original_train = runner.train.train

    def train(*values, **kwargs):
        if (kwargs["num_envs"], kwargs["num_minibatches"], kwargs["num_updates_per_batch"]) != (32, 4, 4):
            raise ValueError("Temporal comparison assumes the frozen 32-env PPO batch")
        kwargs.update(options)
        metadata = dict(options, control_seconds_per_sequence=.05 * args.rollout_length,
            transitions_per_batch=2560, transitions_per_minibatch=640,
            gradient_updates_per_batch=16, gae_lambda=.95, discount=.99,
            source_sha256=digest(output / "sources" / Path(__file__).name))
        (output / "temporal_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return original_train(*values, **kwargs)

    if "--ppo-discount" in rest:
        parser.error("Keep gamma fixed for this isolated temporal comparison")
    runner.train.train = train
    sys.argv = [sys.argv[0], *rest]
    try:
        constrained.main()
    finally:
        runner.train.train = original_train


if __name__ == "__main__":
    main()
