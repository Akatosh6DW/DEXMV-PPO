"""Regression tests for actor-mean preservation and BC teacher anchoring."""

from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jp
import numpy as np

from run_s09_bc_trust import with_std, trusted_loss
from train_orca_matched import initialize_matched


class TrustTests(unittest.TestCase):
    def test_std_changes_sampling_not_mean_or_critic(self):
        net, params = initialize_matched(SimpleNamespace(arm="residual"), 41, .3)
        updated = with_std(params, .03)
        obs = {"state": jax.random.normal(jax.random.PRNGKey(5), (20,123))}
        old = net.policy_network.apply(params[0], params[1], obs)
        new = net.policy_network.apply(updated[0], updated[1], obs)
        np.testing.assert_array_equal(old[:,:17], new[:,:17])
        np.testing.assert_allclose(jax.nn.softplus(new[:,17:])+.001, .03, atol=1e-7)
        for a,b in zip(jax.tree_util.tree_leaves(params[2]),jax.tree_util.tree_leaves(updated[2])):
            np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(jax.nn.softplus(old[:,17:])+.001, .3, atol=1e-7)

    def test_teacher_kl_zero_at_initialization_positive_after_change(self):
        net, teacher = initialize_matched(SimpleNamespace(arm="residual"), 41, .03)
        obs = {"state": jp.ones((4,8,123)), "privileged_state": jp.ones((4,8,140))}
        data = SimpleNamespace(observation=obs)
        fn = trusted_loss(lambda *a, **k: (jp.float32(0),{}), teacher, 1.)
        value, _ = fn(SimpleNamespace(policy=teacher[1]),teacher[0],data,jax.random.PRNGKey(1),ppo_network=net)
        self.assertAlmostEqual(float(value),0.,places=6)
        changed = with_std(teacher,.06)
        value, _ = fn(SimpleNamespace(policy=changed[1]),teacher[0],data,jax.random.PRNGKey(1),ppo_network=net)
        self.assertGreater(float(value),0.)


if __name__ == "__main__":
    unittest.main()
