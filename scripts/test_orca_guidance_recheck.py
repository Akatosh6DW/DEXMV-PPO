"""Focused regression checks for the new experiment, without physics compilation."""

import unittest
from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np

from orca_guidance_recheck import RewardAblationGrasp, make_evaluator, task_reward
from train_orca_guidance_recheck import curve_summary
from train_orca_residual import initialize_zero_mean


class FakeData(NamedTuple):
    qpos: object
    qvel: object
    ctrl: object
    mocap_pos: object


class FakeState(NamedTuple):
    obs: object
    data: object
    metrics: object
    info: object
    reward: object
    done: object


class TerminalEnv:
    def reset(self, key):
        del key
        obs = dict(state=jp.zeros(113), privileged_state=jp.zeros(130))
        data = FakeData(jp.zeros(1), jp.zeros(1), jp.zeros(17), jp.zeros((1, 3)))
        metrics = {k: jp.float32(0) for k in ("hold_sec", "forbidden_contact", "contact_count",
                                             "thumb_contact", "released", "cube_z")}
        info = {k: jp.zeros(17) for k in ("video_base", "requested_residual", "effective_residual", "desired_ctrl")}
        info["cube_spawn_pos"] = jp.zeros(3)
        return FakeState(obs, data, metrics, info, jp.float32(0), jp.float32(0))

    def step(self, state, action):
        del action
        return state._replace(reward=jp.where(state.done > 0, 999., -7.), done=jp.float32(1))


class RecheckTests(unittest.TestCase):
    def test_reward_removes_shaping_only(self):
        np.testing.assert_allclose(task_reward(jp.array([0, 1, 0]), jp.array([0, 0, 1]), .05),
                                   [0, .75, -2.5])

    def test_starting_advantage_not_learning_gain(self):
        result = curve_summary([dict(step=0, success30=.8), dict(step=100, success30=.8)])
        self.assertEqual(result["learning_gain"], 0)
        self.assertAlmostEqual(result["gain_auc"], 0)
        self.assertTrue(result["starting75"])
        self.assertEqual(result["sustained75_step"], 0)

    def test_failed_threshold_is_censored(self):
        result = curve_summary([dict(step=0, success30=0), dict(step=100, success30=.9)])
        self.assertTrue(result["censored"])
        self.assertIsNone(result["sustained75_step"])
        self.assertAlmostEqual(result["auc"], .45)

    def test_reward_variants_preserve_model_and_reset(self):
        a, b = RewardAblationGrasp("pure", "shaped"), RewardAblationGrasp("pure", "hold_only")
        for name in ("body_mass", "body_inertia", "geom_friction", "actuator_ctrlrange"):
            np.testing.assert_array_equal(getattr(a.mj_model, name), getattr(b.mj_model, name))
        self.assertEqual(a._config.to_json(), b._config.to_json())
        self.assertEqual(jax.eval_shape(a.reset, jax.random.PRNGKey(0)).obs["state"].shape, (113,))

    def test_terminal_reward_counted_once_and_sampling_keys_advance(self):
        net, params = initialize_zero_mean(91, .7)
        trace = jax.device_get(make_evaluator(TerminalEnv(), net, [4000], stochastic=True)(params))
        self.assertEqual(float(trace["reward"].sum()), -7.)
        self.assertEqual(int(trace["active"].sum()), 1)
        self.assertFalse(trace["alive"].any())
        self.assertFalse(np.array_equal(trace["action"][0, 0], trace["action"][0, 1]))


if __name__ == "__main__":
    unittest.main()
