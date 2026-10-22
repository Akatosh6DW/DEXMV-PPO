import unittest

import jax
import numpy as np

from probe_s09_feedback_perturbation import ROOT, impulse_state, compare_trace
from run_s09_bc_conditioned import ModeConditionedGrasp
from analyze_s09_feedback_perturbation import paired_rows


class FeedbackPerturbationTests(unittest.TestCase):
    def test_impulse_changes_only_velocity_and_observation(self):
        env = ModeConditionedGrasp(ROOT/"artifacts/orca_benchmark_suite/S09/inputs/model.xml", "pure")
        state = jax.device_get(jax.jit(env.reset)(jax.random.PRNGKey(90523)))
        self.assertIs(impulse_state(env, state, [0,0,0]), state)
        changed = impulse_state(env, state, [.02,0,0])
        np.testing.assert_array_equal(changed.data.qpos, state.data.qpos)
        expected = np.asarray(state.data.qvel).copy()
        expected[env.dof] += .02
        np.testing.assert_allclose(changed.data.qvel, expected, atol=2e-9, rtol=0)
        for key in state.info:
            np.testing.assert_array_equal(changed.info[key], state.info[key])
        mask = np.ones(123, dtype=bool)
        mask[71:74] = False
        np.testing.assert_array_equal(np.asarray(changed.obs["state"])[mask], state.obs["state"][mask])
        np.testing.assert_allclose(changed.obs["state"][71]-state.obs["state"][71], .04, atol=1e-8)
        with self.assertRaises(ValueError):
            impulse_state(env, state, [np.nan,0,0])

    def test_differences_only_use_shared_live_frames(self):
        base = dict(qpos=np.zeros((80,3)), qvel=np.zeros((80,3)), active=np.ones(80,bool))
        altered = dict(qpos=np.ones((80,3)), qvel=np.ones((80,3)), active=np.arange(80)<10)
        result = compare_trace(altered, base, 0, 0)
        self.assertEqual(result["shared_active_frames"], 10)
        self.assertIsNone(result["position_deviation_1s_m"])
        self.assertIsNone(result["position_deviation_4s_m"])
        self.assertAlmostEqual(result["max_position_deviation_m"], np.sqrt(3))

    def test_pairing_does_not_impute_missing_horizons(self):
        row = dict(run="test", seed=1, impulse="x_pos", replay=False,
            nominal_alive_at_end=False, success30=False, hold_sec=0,
            max_released_penetration_m=0, position_deviation_1s_m=.01,
            position_deviation_4s_m=None)
        other = dict(row, replay=True, position_deviation_1s_m=.005, position_deviation_4s_m=.02)
        pair = paired_rows([row, other])[0]
        self.assertFalse(pair["nominal_full_horizon"])
        self.assertEqual(pair["feedback_minus_replay_1s_mm"], 5.)
        self.assertIsNone(pair["feedback_minus_replay_4s_mm"])
        with self.assertRaises(ValueError):
            paired_rows([row,row,other])


if __name__ == "__main__":
    unittest.main()
