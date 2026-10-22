"""R1 must change rewards only, not transitions or the grasp acceptance rule."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import jax
import jax.numpy as jp
import numpy as np

from orca_s09_env import OrcaS09Grasp
from run_s09_reward_job import PenetrationPenaltyGrasp
from s09_suspension import scene_xml


class RewardTest(unittest.TestCase):
    def test_identical_physics_and_explicit_penalty(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"model.xml"
            path.write_text(scene_xml())
            base=OrcaS09Grasp(path)
            revised=PenetrationPenaltyGrasp(path)
            a,b=base.reset(jax.random.PRNGKey(9031)),revised.reset(jax.random.PRNGKey(9031))
            step_a,step_b=jax.jit(base.step),jax.jit(revised.step)
            for _ in range(12):
                a,b=step_a(a,jp.zeros(17)),step_b(b,jp.zeros(17))
                np.testing.assert_array_equal(a.data.integration_words,b.data.integration_words)
                self.assertEqual(float(a.metrics["success30"]),float(b.metrics["success30"]))
                expected=float(jp.clip((a.data.audit[6]-.0015)/.0015,0,4)*5*.05)
                self.assertAlmostEqual(float(a.reward-b.reward),expected,places=5)
            self.assertEqual(revised._config.reward_version,"S09-R1_penetration")


if __name__=="__main__":
    unittest.main()
