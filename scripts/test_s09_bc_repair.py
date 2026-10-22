"""Check the optional training-only BC anchor without simulator changes."""

from types import SimpleNamespace
from pathlib import Path
import unittest

import jax
import jax.numpy as jp
import numpy as np

from run_s09_bc_repair import anchored_loss
from orca_s09_env import OrcaS09Grasp
from orca_s09_bc_observation import RepairObservationGrasp


class AnchorTests(unittest.TestCase):
    def test_observation_changes_leave_physics_unchanged(self):
        model = Path(__file__).resolve().parents[1]/"artifacts/orca_benchmark_suite/S09/inputs/model.xml"
        original = OrcaS09Grasp(model)
        for mode in ("no_history", "initial_axis"):
            changed = RepairObservationGrasp(model, observation_mode=mode)
            a = original.reset(jax.random.PRNGKey(918))
            b = changed.reset(jax.random.PRNGKey(918))
            initial_axis = np.asarray(a.obs["state"][74:77])
            for _ in range(3):
                a = original.step(a, jp.zeros(17))
                b = changed.step(b, jp.zeros(17))
                np.testing.assert_array_equal(a.data.integration_words, b.data.integration_words)
                self.assertEqual(float(a.reward), float(b.reward))
                if mode == "no_history":
                    np.testing.assert_array_equal(b.obs["state"][34:68], 0)
                else:
                    np.testing.assert_array_equal(b.obs["state"][95:98], initial_axis)
                np.testing.assert_array_equal(b.obs["privileged_state"][:123], b.obs["state"])

    def test_balance_and_no_validation_leak(self):
        def apply(_norm, policy, observations):
            return jp.ones((len(observations["state"]), 34)) * policy
        net = SimpleNamespace(policy_network=SimpleNamespace(apply=apply))
        obs = jp.zeros((2, 123))
        early, late = jp.zeros((2, 17)), jp.full((200, 17), .5)
        arrays = (obs, early, jp.zeros((200, 123)), late, obs, jp.full((2, 17), jp.nan))
        original = lambda *a, **k: (jp.float32(2), {"base": jp.float32(2)})
        fn = anchored_loss(original, arrays, 3.)
        value, metrics = jax.jit(lambda p: fn(SimpleNamespace(policy=p), None, None,
                    jax.random.PRNGKey(1), ppo_network=net))(jp.float32(0))
        expected = .5 * float(jp.arctanh(.5)) ** 2
        self.assertAlmostEqual(float(metrics["demo_loss"]), expected, places=6)
        self.assertAlmostEqual(float(value), 2 + 3 * expected, places=6)
        grad = jax.grad(lambda p: fn(SimpleNamespace(policy=p), None, None,
                    jax.random.PRNGKey(1), ppo_network=net)[0])(jp.float32(0))
        self.assertLess(float(grad), 0)

    def test_zero_weight_preserves_ppo_objective(self):
        net = SimpleNamespace(policy_network=SimpleNamespace(
            apply=lambda norm, p, obs: jp.zeros((len(obs["state"]), 34))))
        obs, action = jp.zeros((2, 123)), jp.zeros((2, 17))
        fn = anchored_loss(lambda *a, **k: (jp.float32(7), {}), (obs, action, obs, action), 0)
        value, _ = fn(SimpleNamespace(policy=None), None, None, jax.random.PRNGKey(0), ppo_network=net)
        np.testing.assert_equal(float(value), 7)


if __name__ == "__main__":
    unittest.main()
