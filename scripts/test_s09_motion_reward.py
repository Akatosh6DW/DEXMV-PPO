"""Motion shaping cannot change task dynamics or success labels."""

import unittest

import jax
import jax.numpy as jp
import numpy as np

from run_s09_motion_reward import MotionGrasp, motion_cost
from orca_s09_env import OrcaS09Grasp
from s09_suspension import OUT


class MotionTests(unittest.TestCase):
    def test_gate_cap_and_zero_velocity(self):
        zero = jp.zeros(3)
        large = jp.ones(3)
        self.assertEqual(float(motion_cost(large, large, False, 5)), 0)
        self.assertEqual(float(motion_cost(zero, zero, True, 5)), 0)
        self.assertAlmostEqual(float(motion_cost(large, large, True, 5)), -.25)
        self.assertAlmostEqual(float(motion_cost(jp.array([.0025, 0, 0]), zero, True, 5)), -.03125)

    def test_actual_released_grasp_changes_only_reward(self):
        options = dict(arm="residual", reference=OUT / "inputs/reference.npz", orientation_mode="web")
        base = OrcaS09Grasp(OUT / "inputs/model.xml", **options)
        changed = MotionGrasp(OUT / "inputs/model.xml", **options)
        a, b = base.reset(jax.random.PRNGKey(130001)), changed.reset(jax.random.PRNGKey(130001))
        fa, fb = jax.jit(base.step), jax.jit(changed.step)
        nonzero = False
        for _ in range(85):
            a, b = fa(a, jp.zeros(17)), fb(b, jp.zeros(17))
            np.testing.assert_array_equal(a.data.integration_words, b.data.integration_words)
            for key in a.obs:
                np.testing.assert_array_equal(a.obs[key], b.obs[key])
            for key in a.metrics:
                self.assertEqual(float(a.metrics[key]), float(b.metrics[key]))
            self.assertEqual(float(a.done), float(b.done))
            delta = float(b.metrics["valid_motion_cost"])
            self.assertGreaterEqual(delta, -.25)
            self.assertLessEqual(delta, 0)
            self.assertAlmostEqual(float(b.reward-a.reward), delta, places=6)
            nonzero |= delta < 0
        self.assertTrue(nonzero, "Cover nonzero motion in an actual valid released grasp")


if __name__ == "__main__":
    unittest.main()
