"""Versioned open-hand cube acquisition task; historical F01 stays untouched."""

import jax.numpy as jp

from orca_native_physics import OrcaNativeMatchedGrasp
from orca_refit_env import BASE_WEIGHTS


class OrcaAcquisitionGrasp(OrcaNativeMatchedGrasp):
    def __init__(self, model, arm, reference, *, jitter=.002, shaping=1., **kwargs):
        super().__init__(model, arm, reference, dict(BASE_WEIGHTS), tactile=True, **kwargs)
        if not 0 <= jitter <= .006 or not 0 <= shaping <= 3:
            raise ValueError("Invalid development task configuration")
        self._config.spawn_config.cube_jitter = [jitter, jitter, 0.]
        self.shaping = shaping

    def reset(self, rng):
        state = super().reset(rng)
        return state.replace(metrics=dict(state.metrics, acquisition_reward=jp.float32(0)))

    def step(self, state, action):
        nxt = super().step(state, action)
        m = nxt.metrics
        pair = (m["thumb_contact"] > 0) & (m["contact_count"] >= 2)
        # Shared task feedback, independent of demonstrations or reference angles.
        acquisition = jp.where(nxt.info["step"] * self.dt < 3.,
                               3 * m["proximity"] + 2 * m["thumb_contact"] + 2 * pair, 0.)
        return nxt.replace(reward=nxt.reward + self.shaping * acquisition * self.dt,
                           metrics=dict(m, acquisition_reward=acquisition))
