"""Change only GAE lambda on the existing 80-step temporal PPO recipe.

Use Brax's existing GAE implementation. Longer credit propagation is not a new
reward, a longer rollout, or an unbootstrapped whole-episode Monte Carlo return.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import run_s09_temporal_ppo as temporal
from retarget_s09_reference import digest


def credit_options(value):
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("GAE lambda must be finite and in [0,1]")
    return dict(gae_lambda=float(value))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--credit-lambda", type=float, default=1.)
    args, rest = parser.parse_known_args()
    options = credit_options(args.credit_lambda)
    if "--rollout-length" not in rest or rest[rest.index("--rollout-length")+1] != "80":
        parser.error("Compare with the existing 80-step temporal controls")
    if "--teacher-kl" in rest or "--ppo-discount" in rest:
        parser.error("Keep teacher loss and gamma unchanged")
    output = Path(rest[rest.index("--output")+1])
    runner = temporal.constrained.improvement.runner
    old_train = runner.train.train

    def train(*values, **kwargs):
        if kwargs["unroll_length"] != 80 or kwargs["discounting"] != .99:
            raise ValueError("Unexpected temporal control recipe")
        kwargs.update(options)
        path = output/"temporal_config.json"
        metadata = json.loads(path.read_text())
        metadata.update(options)
        path.write_text(json.dumps(metadata, indent=2)+"\n")
        (output/"credit_config.json").write_text(json.dumps(dict(
            options, reward_version="S09-R0 unchanged", discount=.99,
            unroll_length=80, bootstrap_at_fragment_end=True,
            inference_changed=False, source_sha256=digest(output/"sources"/Path(__file__).name)),
            indent=2)+"\n")
        return old_train(*values, **kwargs)

    runner.train.train = train
    sys.argv = [sys.argv[0], *rest]
    try:
        temporal.main()
    finally:
        runner.train.train = old_train


if __name__ == "__main__":
    main()
