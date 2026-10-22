"""Value targets preserve failures and do not include padded future rewards."""

import unittest

import jax
import jax.numpy as jp
import numpy as np

from pretrain_s09_critic import collector,returns_to_go
import run_s09_job as runner
from run_s09_bc_conditioned import ModeConditionedGrasp
from s09_suspension import OUT


class CriticPretrainingTests(unittest.TestCase):
    def test_finite_episode_return_and_padding(self):
        rewards=np.array([[1,2,999],[1,-3,999]],np.float32)
        active=np.array([[1,1,0],[1,1,0]],bool)
        np.testing.assert_allclose(returns_to_go(rewards,active,.5),[[2,2,0],[-.5,-3,0]])

    def test_collector_matches_frozen_evaluator(self):
        env=ModeConditionedGrasp(OUT/"inputs/model.xml")
        net,params=runner.initialize_matched(env,3,.03)
        seeds=jp.array([140000],jp.int32)
        initial,expected=jax.device_get(runner.evaluator(env,net)(params,seeds))
        actual_initial,actual=jax.device_get(collector(env,net)(params,seeds))
        np.testing.assert_array_equal(initial,actual_initial)
        for key in ("action","reward","active"):
            np.testing.assert_array_equal(expected[key],actual[key])
        np.testing.assert_array_equal(expected["obs"],actual["privileged_obs"][...,:123])
        self.assertEqual(actual["privileged_obs"].shape[-1],140)


if __name__ == "__main__":
    unittest.main()
