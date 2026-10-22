"""Use Brax's existing KL-adaptive PPO learning rate for BC fine-tuning."""

import json
from pathlib import Path
import sys

from brax.training.agents.ppo.optimizer import LRSchedule
import run_s09_bc_trust as trust


def main():
    original = trust.repair.runner.train.train
    output = Path(sys.argv[sys.argv.index("--output")+1])
    def adaptive(*args, **kwargs):
        kwargs.update(learning_rate_schedule=LRSchedule.ADAPTIVE_KL, desired_kl=.01)
        return original(*args, **kwargs)
    trust.repair.runner.train.train = adaptive
    try:
        trust.main()
    finally:
        if output.exists():
            (output/"adaptive_config.json").write_text(json.dumps(dict(
                learning_rate_schedule="ADAPTIVE_KL", desired_kl=.01,
                implementation="Installed Brax PPO optimizer; adjusts LR from on-policy old/new KL",
                distinction="Not the frozen BC teacher KL loss"), indent=2)+"\n")


if __name__ == "__main__":
    main()
