import unittest

import jax
import jax.numpy as jp
import numpy as np

import run_s09_job as runner
from run_s09_bc_conditioned import ModeConditionedGrasp
from run_s09_disturbance_training import DisturbanceTraining


class DisturbanceTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = runner.ROOT/"artifacts/orca_benchmark_suite/S09"
        cls.env = ModeConditionedGrasp(root/"inputs/model.xml", "pure")
        with np.load(root/"demonstrations/trajectories.npz") as z:
            cls.actions = jp.asarray(z["action"][0,:41])
        def prefix(key):
            return jax.lax.scan(lambda s,a:(cls.env.step(s,a),None),
                cls.env.reset(key), cls.actions[:39])[0]
        cls.state = jax.device_get(jax.jit(prefix)(jax.random.PRNGKey(90504)))

    def test_zero_probability_preserves_all_original_state(self):
        wrapped = DisturbanceTraining(self.env, 0)
        expected = jax.jit(self.env.step)(self.state, self.actions[39])
        actual = jax.jit(wrapped.step)(self.state, self.actions[39])
        self.assertEqual(float(actual.metrics["training_kick"]), 0)
        actual = actual.replace(metrics={k:v for k,v in actual.metrics.items() if k!="training_kick"})
        for a,b in zip(jax.tree_util.tree_leaves(expected),jax.tree_util.tree_leaves(actual)):
            np.testing.assert_array_equal(a,b)

    def test_one_kick_preserves_phase_position_reward_and_counters(self):
        wrapped = DisturbanceTraining(self.env, 1)
        expected = jax.jit(self.env.step)(self.state, self.actions[39])
        actual = jax.jit(wrapped.step)(self.state, self.actions[39])
        self.assertEqual(float(actual.metrics["training_kick"]), 1)
        np.testing.assert_array_equal(actual.data.qpos, expected.data.qpos)
        self.assertEqual(float(actual.reward), float(expected.reward))
        delta = np.asarray(actual.data.qvel)-np.asarray(expected.data.qvel)
        self.assertAlmostEqual(float(np.linalg.norm(delta)), .02, places=6)
        outside = np.ones(len(delta),bool)
        outside[self.env.dof:self.env.dof+3] = False
        np.testing.assert_array_equal(delta[outside], 0)
        for key in expected.info:
            np.testing.assert_array_equal(actual.info[key], expected.info[key])
        mask = np.ones(123,bool)
        mask[71:74] = False
        np.testing.assert_array_equal(np.asarray(actual.obs["state"])[mask],np.asarray(expected.obs["state"])[mask])
        nxt = jax.jit(wrapped.step)(actual, self.actions[40])
        self.assertEqual(float(nxt.metrics["training_kick"]), 0)

    def test_batched_callback_matches_individual_kicks(self):
        wrapped = DisturbanceTraining(self.env, 1)
        words = np.asarray(self.state.data.integration_words)
        velocity = np.asarray(self.state.data.qvel)
        delta = np.array([[0,0,0],[.02,0,0],[0,0,-.02]])
        outputs = wrapped._batch(np.tile(words,(3,1)), np.tile(velocity,(3,1)), delta)
        for i in range(3):
            expected = wrapped._single(words,velocity,delta[i])
            for a,b in zip(outputs,expected):
                np.testing.assert_array_equal(a[i],b)
        with self.assertRaises(ValueError):
            DisturbanceTraining(self.env, 1.1)


if __name__ == "__main__":
    unittest.main()
