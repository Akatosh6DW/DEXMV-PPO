"""Paired tactile observations for the source-matched grasp experiment.

These are simulated distributed finger force sensors, not measured hardware
taxels. No target-object identity is used to select sensor contacts.
"""

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from orca_hic53_env import OrcaHic53Grasp
from orca_contact_forces import normal_loads
from orca_refit_env import BASE_WEIGHTS

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
HOLD_COUNT_OBS_INDEX = 94


def reference_variant(reference, initial, mode, max_delta=.1):
    """Control-only ablation: retain the same physical reset and joint limits."""
    reference = np.asarray(reference)
    initial = np.asarray(initial)
    if mode not in ("video", "endpoint", "initial"):
        raise ValueError("Unknown reference mode")
    if mode == "video":
        return reference.copy()
    if mode == "initial":
        return np.broadcast_to(initial, reference.shape).copy()
    # A slew-limited direct path, not a replay of the video's intermediate poses.
    difference = reference[-1] - initial
    distance = np.arange(1, len(reference) + 1)[:, None] * max_delta
    return initial + np.clip(difference, -distance, distance)


def height_retention(cube_z, spawn_z, released, paired, alive, xp=jp):
    """Bounded feedback for incipient slip, only after unsupported acquisition."""
    return xp.where(released & paired & alive,
                    xp.exp(-xp.maximum(spawn_z - cube_z, 0.) / .005), 0.)


def finger_normal_forces(geom, address, efc_force, membership, xp=np, cone="pyramidal"):
    """Sum solver-backed normal loads over finger sensor regions."""
    geom = xp.asarray(geom)
    membership = xp.asarray(membership)
    normal = normal_loads(address, efc_force, cone, xp=xp)
    normal = xp.where(xp.all(geom >= 0, axis=1), normal, 0.)
    belongs = membership[:, xp.maximum(geom[:, 0], 0)] | membership[:, xp.maximum(geom[:, 1], 0)]
    return xp.sum(belongs * normal[None, :], axis=1)


def touch_observation(delayed_force, enabled, scale=5., contact_threshold=.02, xp=np):
    sensed = xp.clip(delayed_force, 0., scale)
    vector = xp.concatenate([sensed / scale, (sensed >= contact_threshold).astype(xp.float32)])
    return xp.where(enabled, vector, xp.zeros_like(vector))


class OrcaMatchedGrasp(OrcaHic53Grasp):
    actor_obs_dim = 123
    critic_obs_dim = 140

    def __init__(self, model, arm, reference, weights=None, speed_scale=.05,
                 tactile=False, tactile_noise=.02, tactile_filter=.5, residual_bound=.08,
                 reference_mode="video"):
        if reference_mode not in ("video", "endpoint", "initial"):
            raise ValueError("Unknown reference mode")
        if arm != "residual" and reference_mode != "video":
            raise ValueError("Reference ablations require residual control")
        self.reference_mode = reference_mode
        if not np.isfinite(residual_bound) or not 0 < residual_bound <= .15:
            raise ValueError("Matched residual bound must be in (0, .15] radians")
        if not np.isfinite([tactile_noise, tactile_filter]).all() or tactile_noise < 0 or not 0 < tactile_filter <= 1:
            raise ValueError("Invalid tactile noise/filter")
        self.tactile_enabled = bool(tactile)
        self.tactile_noise = float(tactile_noise)
        self.tactile_filter = float(tactile_filter)
        weights = dict(BASE_WEIGHTS if weights is None else weights)
        height_weight = weights.pop("height_retention", None)
        if height_weight is not None and (not np.isfinite(height_weight) or height_weight < 0):
            raise ValueError("Height retention weight must be nonnegative and finite")
        super().__init__(model, arm, reference, weights, speed_scale)
        self.residual_bound = float(residual_bound)
        if arm == "residual" and reference_mode != "video":
            self.reference_ctrl = jp.asarray(reference_variant(
                self.reference_ctrl, self._config.initial_ctrl, reference_mode,
                self._config.max_ctrl_delta))
        if height_weight is not None:
            self.weights["height_retention"] = height_weight
        physical = self.mj_model.geom_contype != 0
        if not np.all(self.mj_model.geom_condim[physical] == 4):
            raise ValueError("Force decoder requires condim=4 contacts")
        mask = np.zeros((5, self.mj_model.ngeom), dtype=bool)
        for i, finger in enumerate(FINGERS):
            mask[i, self._finger_contact_gids[finger]] = True
        self.sensor_membership = jp.asarray(mask)

    def _forces(self, data):
        return finger_normal_forces(data.contact.geom, data.contact.efc_address, data.efc_force,
                                    self.sensor_membership, xp=jp, cone=self.contact_cone)

    def _obs(self, data, info):
        base = super()._obs(data, info)
        # The legacy clean-hold counter depends on privileged contact flags.
        # Do not leak it to the supposedly no-tactile actor; critic is shared.
        base["state"] = base["state"].at[HOLD_COUNT_OBS_INDEX].set(0.)
        touch = info.get("touch_obs", jp.zeros(10))
        critic_touch = info.get("critic_touch_obs", jp.zeros(10))
        return dict(state=jp.concatenate([base["state"], touch]),
                    privileged_state=jp.concatenate([base["privileged_state"], critic_touch]))

    def reset(self, rng):
        state = super().reset(rng)
        info = dict(state.info, touch_rng=jax.random.fold_in(rng, 43921),
                    touch_filter=jp.zeros(5), touch_obs=jp.zeros(10), critic_touch_obs=jp.zeros(10),
                    tactile_actor_mask=jp.asarray(self.tactile_enabled),
                    finger_normal_force=self._forces(state.data))
        metrics = dict(state.metrics, finger_force_max=jp.float32(0), finger_force_sum=jp.float32(0))
        if "height_retention" in self.weights:
            metrics["component_height_retention"] = jp.float32(0)
        return state.replace(info=info, metrics=metrics, obs=self._obs(state.data, info))

    def step(self, state, action):
        nxt = super().step(state, action)
        rng, noise_key = jax.random.split(state.info["touch_rng"])
        force = self._forces(nxt.data)
        noisy = jp.clip(force + self.tactile_noise * jax.random.normal(noise_key, (5,)), 0., 5.)
        filtered = (1 - self.tactile_filter) * state.info["touch_filter"] + self.tactile_filter * noisy
        # One control-step delay. Both ablations execute the same sensor/RNG path.
        sensed = touch_observation(state.info["touch_filter"], state.info["tactile_actor_mask"], xp=jp)
        critic_sensed = touch_observation(state.info["touch_filter"], True, xp=jp)
        info = dict(nxt.info, touch_rng=rng, touch_filter=filtered, touch_obs=sensed,
                    critic_touch_obs=critic_sensed, finger_normal_force=force)
        metrics = dict(nxt.metrics, finger_force_max=jp.max(force), finger_force_sum=jp.sum(force))
        reward = nxt.reward
        if "height_retention" in self.weights:
            m = nxt.metrics
            paired = (m["thumb_contact"] > 0) & (m["contact_count"] >= 2) & (m["forbidden_contact"] == 0)
            value = height_retention(m["cube_z"], nxt.info["cube_spawn_pos"][2],
                                     m["released"] > 0, paired, nxt.done == 0)
            metrics["component_height_retention"] = value
            reward = reward + self.weights["height_retention"] * value * self.dt
        return nxt.replace(info=info, metrics=metrics, reward=reward, obs=self._obs(nxt.data, info))
