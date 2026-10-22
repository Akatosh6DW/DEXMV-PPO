"""Load margin shaping cannot change the task or incentivize unbounded force."""

import unittest

import jax
import jax.numpy as jp
import numpy as np

from run_s09_contact_margin import MarginGrasp,load_coverage
from orca_s09_env import OrcaS09Grasp
from s09_suspension import OUT


class MarginTests(unittest.TestCase):
    def test_requires_thumb_and_opposition_and_saturates(self):
        self.assertEqual(float(load_coverage(jp.array([0,1,1,1,1]),.2)),0)
        self.assertEqual(float(load_coverage(jp.array([1,0,0,0,0]),.2)),0)
        self.assertAlmostEqual(float(load_coverage(jp.array([.02,.02,0,0,0]),.2)),.01)
        self.assertEqual(float(load_coverage(jp.array([.2,.2,0,0,0]),.2)),1)
        self.assertEqual(float(load_coverage(jp.array([100,100,0,0,0]),.2)),1)

    def test_real_rollout_physics_and_success_unchanged(self):
        kwargs=dict(arm="residual",reference=OUT/"inputs/reference.npz")
        aenv=OrcaS09Grasp(OUT/"inputs/model.xml",**kwargs)
        benv=MarginGrasp(OUT/"inputs/model.xml",**kwargs)
        a,b=aenv.reset(jax.random.PRNGKey(130000)),benv.reset(jax.random.PRNGKey(130000))
        fa,fb=jax.jit(aenv.step),jax.jit(benv.step)
        changed=False
        for _ in range(120):
            a,b=fa(a,jp.zeros(17)),fb(b,jp.zeros(17))
            np.testing.assert_array_equal(a.data.integration_words,b.data.integration_words)
            np.testing.assert_array_equal(a.obs["state"],b.obs["state"])
            self.assertEqual(float(a.metrics["success30"]),float(b.metrics["success30"]))
            self.assertEqual(float(a.done),float(b.done))
            delta=float(b.metrics["coverage_reward_delta"])
            self.assertLessEqual(delta,0)
            self.assertAlmostEqual(float(b.reward-a.reward),delta,places=6)
            changed |= delta < -1e-6
        self.assertTrue(changed,"Must exercise a different reward, not merely identical zero contact")


if __name__ == "__main__":
    unittest.main()
