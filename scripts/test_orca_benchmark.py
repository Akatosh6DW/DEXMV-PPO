"""Focused invariants for the separate Orca comparison environment."""

from pathlib import Path
from types import SimpleNamespace
import unittest

import jax.numpy as jp
import numpy as np

from check_orca_benchmark import hic_closure_phase
from orca_benchmark_env import OrcaBenchmark, default_config
from evaluate_orca_policy import longest_run
from train_orca_comparison import initialize, make_demo_regularized_loss


class BenchmarkTests(unittest.TestCase):
    def test_no_demo_and_long_enough(self):
        cfg = default_config()
        self.assertFalse(cfg.residual_prior.enabled)
        self.assertGreaterEqual(cfg.episode_length * cfg.ctrl_dt, 35)
        self.assertEqual(cfg.reward_config.hold_success_sec, 30)

    def test_contact_requires_force_and_nonpositive_gap(self):
        env = OrcaBenchmark.__new__(OrcaBenchmark)
        env._cube_geom_id = 1
        def hit(distance, force, address=0):
            contact = SimpleNamespace(dist=jp.array([distance]),
                                      geom=jp.array([[1, 2]]), efc_address=jp.array([address]))
            data = SimpleNamespace(contact=contact, efc_force=jp.ones(6) * force)
            return float(env._has_cube_contact(data, [2]))
        self.assertEqual(hit(-.001, 1), 1)
        self.assertEqual(hit(.0001, 1), 0)
        self.assertEqual(hit(-.001, 0), 0)
        self.assertEqual(hit(-.001, 1, -1), 0)

    def test_hic_phase_is_bounded_monotone(self):
        phase = hic_closure_phase(Path("artifacts/cube20/hic20/prepared/human_cube.npz"))
        self.assertEqual(len(phase), 61)
        self.assertEqual(phase[0], 0)
        self.assertEqual(phase[-1], 1)
        self.assertTrue(np.all(np.diff(phase) >= 0))

    def test_control_bounds(self):
        env = OrcaBenchmark.__new__(OrcaBenchmark)
        env._lowers = jp.array([-1., -.2])
        env._uppers = jp.array([.5, 2.])
        np.testing.assert_allclose(env.ctrl_to_action(env._lowers), [-1, -1])
        np.testing.assert_allclose(env.ctrl_to_action(env._uppers), [1, 1])

    def test_continuous_not_accumulated_hold(self):
        self.assertAlmostEqual(longest_run([True] * 400 + [False] + [True] * 400), 20)
        self.assertAlmostEqual(longest_run([True] * 600), 30)

    def test_explicit_exploration_scale(self):
        obs = dict(state=jp.zeros(95), privileged_state=jp.zeros(112))
        env = SimpleNamespace(action_size=17, reset=lambda key: SimpleNamespace(obs=obs))
        net, params = initialize(env, 10, init_std=.2)
        logits = net.policy_network.apply(params[0], params[1], obs)
        from jax.nn import softplus
        np.testing.assert_allclose(softplus(logits[17:]) + .001, .2, atol=1e-6)

    def test_demo_loss_changes_policy_not_reward(self):
        import jax
        from brax.training.agents.ppo.losses import PPONetworkParams
        obs = dict(state=jp.zeros(95), privileged_state=jp.zeros(112))
        env = SimpleNamespace(action_size=17, reset=lambda key: SimpleNamespace(obs=obs))
        net, params = initialize(env, 10, init_std=.2)
        original = lambda *args, **kwargs: (jp.float32(2), {"original": jp.float32(2)})
        wrapped = make_demo_regularized_loss(original, jp.zeros((4, 95)), jp.ones((4, 17)) * .5, 1)
        value, metrics = wrapped(PPONetworkParams(params[1], params[2]), params[0], None,
                                 jax.random.PRNGKey(0), ppo_network=net)
        self.assertGreater(float(value), 2)
        self.assertGreater(float(metrics["demo_loss"]), 0)
        self.assertEqual(float(metrics["original"]), 2)


if __name__ == "__main__":
    unittest.main()
