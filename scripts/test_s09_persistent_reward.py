"""Persistence shaping must not reward invalid support or modify physics."""

import unittest

import jax
import jax.numpy as jp
import numpy as np

from run_s09_persistent_reward import PersistentGrasp, hold_bonus
from orca_s09_env import OrcaS09Grasp
from s09_suspension import OUT


class PersistentTests(unittest.TestCase):
    def test_invalid_or_tainted_hold_gets_no_bonus(self):
        self.assertEqual(float(hold_bonus(600,False,True,15)),0)
        self.assertEqual(float(hold_bonus(600,True,False,15)),0)
        self.assertEqual(float(hold_bonus(0,True,True,15)),0)
        self.assertAlmostEqual(float(hold_bonus(600,True,True,15)),.75)
        self.assertAlmostEqual(float(hold_bonus(800,True,True,15)),.75)

    def test_continuous_hold_beats_same_duration_fragmented(self):
        one = hold_bonus(jp.arange(1,601),True,True,15).sum()
        two = 2*hold_bonus(jp.arange(1,301),True,True,15).sum()
        self.assertGreater(float(one),float(two))

    def test_reward_only_on_real_reference_rollout(self):
        options=dict(arm="residual",reference=OUT/"inputs/reference.npz",orientation_mode="web")
        base=OrcaS09Grasp(OUT/"inputs/model.xml",**options)
        revised=PersistentGrasp(OUT/"inputs/model.xml",**options)
        a,b=base.reset(jax.random.PRNGKey(130001)),revised.reset(jax.random.PRNGKey(130001))
        f,g=jax.jit(base.step),jax.jit(revised.step)
        positive=False
        for _ in range(85):
            a,b=f(a,jp.zeros(17)),g(b,jp.zeros(17))
            np.testing.assert_array_equal(a.data.integration_words,b.data.integration_words)
            np.testing.assert_array_equal(a.obs["state"],b.obs["state"])
            self.assertEqual(float(a.metrics["success30"]),float(b.metrics["success30"]))
            bonus=float(b.metrics["hold_progress_bonus"])
            self.assertAlmostEqual(float(b.reward-a.reward),bonus,places=6)
            positive |= bonus>0
        self.assertTrue(positive,"Test must cover a valid released grasp, not only the zero-bonus branch")


if __name__ == "__main__":
    unittest.main()
