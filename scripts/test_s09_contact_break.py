"""A contact interruption penalty cannot change the physical success criterion."""

import unittest

import jax
import jax.numpy as jp
import numpy as np

from run_s09_contact_break import BreakGrasp,break_cost
from orca_s09_env import OrcaS09Grasp
from s09_suspension import OUT


class BreakTests(unittest.TestCase):
    def test_only_existing_released_streak_break_is_penalized(self):
        previous=jp.array([0,1,1,1,0])
        released=jp.array([True,False,True,True,True])
        valid=jp.array([False,False,True,False,True])
        np.testing.assert_array_equal(break_cost(previous,released,valid,5),[0,0,0,-5,0])

    def test_real_break_changes_only_reward(self):
        kwargs=dict(arm="residual",reference=OUT/"inputs/reference.npz")
        aenv=OrcaS09Grasp(OUT/"inputs/model.xml",**kwargs)
        benv=BreakGrasp(OUT/"inputs/model.xml",**kwargs)
        a,b=aenv.reset(jax.random.PRNGKey(130001)),benv.reset(jax.random.PRNGKey(130001))
        fa,fb=jax.jit(aenv.step),jax.jit(benv.step)
        events=0
        for _ in range(200):
            a,b=fa(a,jp.zeros(17)),fb(b,jp.zeros(17))
            np.testing.assert_array_equal(a.data.integration_words,b.data.integration_words)
            for key in a.obs:
                np.testing.assert_array_equal(a.obs[key],b.obs[key])
            for key in a.metrics:
                self.assertEqual(float(a.metrics[key]),float(b.metrics[key]))
            self.assertEqual(float(a.done),float(b.done))
            delta=float(b.metrics["contact_break_cost"])
            self.assertIn(delta,(0.,-5.))
            self.assertAlmostEqual(float(b.reward-a.reward),delta,places=6)
            events+=int(delta<0)
        self.assertGreater(events,0,"Exercise an actual physical contact interruption")


if __name__ == "__main__":
    unittest.main()
