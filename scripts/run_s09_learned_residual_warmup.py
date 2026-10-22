"""Fit the inherited critic on current-policy rollouts before residual updates.

Warmup interactions are included in the total online budget. Residual weights
and Adam moments stay fixed during warmup; the frozen learned base never trains.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import run_s09_constrained_ppo as constrained
import run_s09_learned_residual_experts as routed
from run_s09_critic_warmup import warmup_optimizer
from retarget_s09_reference import digest


@contextmanager
def delayed_residual(warmup_steps, output):
    if warmup_steps <= 0 or warmup_steps % 2560:
        raise ValueError("Warmup must be a positive multiple of 2560")
    updates = warmup_steps//2560*16
    original_split = constrained.split_adam
    runner = routed.learned.runner
    original_train = runner.train.train

    def train(*args, **kwargs):
        expected = dict(num_envs=32, batch_size=32, num_minibatches=4,
                        unroll_length=20, num_updates_per_batch=4)
        if any(kwargs[k] != v for k,v in expected.items()) or kwargs.get("action_repeat",1) != 1:
            raise ValueError("Warmup conversion requires the unchanged S09 rollout")
        source = output/"sources"/Path(__file__).name
        record = dict(actor_warmup_transitions=warmup_steps, actor_warmup_gradient_updates=updates,
            total_online_transitions=kwargs["num_timesteps"], warmup_included_in_budget=True,
            actor_and_adam_moments_frozen=True, critic_uses_current_policy_rollouts=True,
            base_unchanged=True, action_law_unchanged=True, reward_version="S09-R0 unchanged",
            source_sha256=digest(source), optimizer_source_sha256=digest(Path(constrained.__file__)),
            delay_source_sha256=digest(Path(sys.modules[warmup_optimizer.__module__].__file__)))
        (output/"residual_warmup_config.json").write_text(json.dumps(record,indent=2)+"\n")
        return original_train(*args,**kwargs)

    constrained.split_adam = lambda actor,critic: warmup_optimizer(actor,critic,updates)
    runner.train.train = train
    try:
        yield
    finally:
        constrained.split_adam = original_split
        runner.train.train = original_train


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--actor-warmup-steps",type=int,required=True)
    args,rest = parser.parse_known_args()
    total = int(rest[rest.index("--steps")+1])
    if not 0 < args.actor_warmup_steps < total or args.actor_warmup_steps % 2560:
        parser.error("Specify a positive 2560-aligned warmup below the total budget")
    if "--checkpoint" in rest:
        parser.error("Use fresh zero residuals, not an already trained residual checkpoint")
    output = Path(rest[rest.index("--output")+1])
    sys.argv = [sys.argv[0],*rest]
    with delayed_residual(args.actor_warmup_steps,output):
        routed.main()


if __name__ == "__main__":
    main()
