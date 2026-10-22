"""Give only the R2 critic the history variables used by its reward.

Actor inputs, deployment actions, physics and success labels are unchanged.
Two formerly zero critic slots encode current hold progress and clean history.
Their input weights start at zero to preserve the original value function.
"""

import json
from pathlib import Path
import sys

import run_s09_persistent_constrained_ppo as persistence
from retarget_s09_reference import digest
import jax
import jax.numpy as jp

HISTORY_ROWS = (77, 94)


def history_features(info):
    clean = ((info["initial_penetration"] <= .001) & (info["max_depth"] <= .003)
             & ~info["ever_forbidden"])
    return clean.astype(jp.float32), jp.minimum(info["hold_steps"]/600., 1.)


class RewardHistoryGrasp(persistence.persistent.PersistentConditionedGrasp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with self._config.unlocked():
            self._config.critic_history_features = "clean_77_hold_progress_94"

    def _obs(self, data, info):
        obs = super()._obs(data, info)
        clean, progress = history_features(info)
        value = obs["privileged_state"].at[77].set(clean).at[94].set(progress)
        return dict(obs, privileged_state=value)


def neutral_history_value(params):
    result = jax.tree_util.tree_map(lambda x: jp.array(x), params)
    head = result[2]["params"]["hidden_0"]
    if head["kernel"].shape[0] != 140:
        raise ValueError("The unchanged S09 critic must have 140 inputs")
    head["kernel"] = head["kernel"].at[jp.asarray(HISTORY_ROWS), :].set(0)
    return result


def main():
    argv = sys.argv[1:]
    if "--ppo-initial-std" in argv:
        raise ValueError("Do not combine critic history with changed actor exploration")
    output = Path(argv[argv.index("--output")+1])
    runner = persistence.constrained.improvement.runner
    old_train = runner.train.train
    old_class = persistence.persistent.PersistentConditionedGrasp

    def train(*values, **kwargs):
        kwargs["restore_params"] = neutral_history_value(kwargs["restore_params"])
        metadata = dict(critic_feature_indices=list(HISTORY_ROWS),
            critic_feature_definitions=["initial_depth<=1mm and max_released_depth<=3mm and no_released_forbidden",
                                        "min(current_continuous_valid_steps/600,1)"],
            corresponding_critic_input_rows_zeroed_at_start=True,
            actor_parameters_and_inputs_unchanged=True,
            history_observable_to_actor=False, future_information_used=False,
            reward_version="S09-R2_qualified_hold_progress unchanged",
            physical_success_criterion_unchanged=True,
            source_sha256=digest(output/"sources"/Path(__file__).name))
        (output/"critic_history_config.json").write_text(json.dumps(metadata, indent=2)+"\n")
        return old_train(*values, **kwargs)

    runner.train.train = train
    persistence.persistent.PersistentConditionedGrasp = RewardHistoryGrasp
    try:
        persistence.main()
    finally:
        runner.train.train = old_train
        persistence.persistent.PersistentConditionedGrasp = old_class


if __name__ == "__main__":
    main()
