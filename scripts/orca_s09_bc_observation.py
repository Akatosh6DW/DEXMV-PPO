"""Observation-only BC diagnostics; no action, dynamics or reward changes."""

import jax.numpy as jp

from orca_s09_env import OrcaS09Grasp


class RepairObservationGrasp(OrcaS09Grasp):
    def __init__(self, *args, observation_mode="original", **kwargs):
        super().__init__(*args, **kwargs)
        if self.arm != "pure" or observation_mode not in ("original", "no_history", "initial_axis"):
            raise ValueError("Repair observation is for full-target BC+PPO only")
        self.observation_mode = observation_mode

    def _obs(self, data, info):
        obs = super()._obs(data, info)
        actor = obs["state"]
        if self.observation_mode == "no_history":
            actor = actor.at[34:68].set(0)
        elif self.observation_mode == "initial_axis":
            actor = actor.at[95:98].set(info.get("initial_axis", actor[74:77]))
        return dict(state=actor, privileged_state=jp.concatenate([actor, data.actuator_force]))

    def reset(self, rng):
        state = super().reset(rng)
        info = dict(state.info, initial_axis=state.obs["state"][74:77])
        return state.replace(info=info, obs=self._obs(state.data, info))
