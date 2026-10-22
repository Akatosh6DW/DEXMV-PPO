"""Learn previously masked feedback connections more slowly during full-target PPO.

The BC phase/mode features, actor head, critic, inference observations, and
action law are unchanged. Scale Adam UPDATES, not gradients that Adam rescales.
"""

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jp
import optax

import run_s09_constrained_ppo as constrained
from run_s09_bc_phase_curriculum import KEEP
from retarget_s09_reference import digest


def scale_feedback_updates(scale):
    if not 0<scale<=1:
        raise ValueError("Keep every feedback input trainable: scale must lie in (0,1]")

    def init(params):
        if params.policy["params"]["hidden_0"]["kernel"].shape[0]!=len(KEEP):
            raise ValueError("Unexpected actor input dimension")
        return optax.EmptyState()

    def update(updates,state,params=None):
        del params

        def rescale(path,value):
            if tuple(getattr(key,"key",None) for key in path)==("params","hidden_0","kernel"):
                return value*jp.where(jp.asarray(KEEP),1.,scale)[:,None]
            return value

        policy=jax.tree_util.tree_map_with_path(rescale,updates.policy)
        return updates.replace(policy=policy),state

    return optax.GradientTransformation(init,update)


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--new-input-scale",type=float,default=.1)
    args,rest=parser.parse_known_args()
    scale_feedback_updates(args.new_input_scale)
    if "--critic-lr" not in rest or rest[rest.index("--method")+1]!="guided":
        parser.error("Use explicit split actor/critic rates and full-target guided policy")
    output=Path(rest[rest.index("--output")+1])
    original_split=constrained.split_adam
    runner=constrained.improvement.runner
    original_train=runner.train.train

    def split(actor_rate,critic_rate):
        return optax.chain(original_split(actor_rate,critic_rate),scale_feedback_updates(args.new_input_scale))

    def train(*values,**kwargs):
        metadata=dict(new_input_update_multiplier=args.new_input_scale,
            unchanged_actor_input_columns=[int(i) for i in KEEP.nonzero()[0]],
            all_actor_weights_trainable=True,inference_changed=False,reference_action_added=False,
            critic_updates_changed=False,reward_version="S09-R0 unchanged",
            source_sha256=digest(output/"sources"/Path(__file__).name))
        (output/"feedback_config.json").write_text(json.dumps(metadata,indent=2)+"\n")
        return original_train(*values,**kwargs)

    constrained.split_adam=split
    runner.train.train=train
    sys.argv=[sys.argv[0],*rest]
    try:
        constrained.main()
    finally:
        constrained.split_adam=original_split
        runner.train.train=original_train


if __name__ == "__main__":
    main()
