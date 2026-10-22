"""Verify options and real Brax delayed-credit propagation, including terminals."""

import unittest

import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo.losses import compute_gae

from run_s09_long_credit_ppo import credit_options


class CreditTests(unittest.TestCase):
    def test_only_lambda_changes(self):
        self.assertEqual(credit_options(1), {"gae_lambda": 1.})
        self.assertEqual(credit_options(.95), {"gae_lambda": .95})
        for value in (-.1, 1.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                credit_options(value)

    def test_delayed_reward_propagates_with_existing_brax_gae(self):
        zero = jp.zeros((80, 1))
        reward = zero.at[60, 0].set(1.)
        outputs = []
        for lam in (.95, 1.):
            values, advantages = compute_gae(zero, zero, reward, zero, jp.zeros(1),
                                             lambda_=lam, discount=.99)
            np.testing.assert_allclose(values[0, 0], (.99*lam)**60, rtol=1e-5)
            self.assertTrue(np.isfinite(np.asarray(advantages)).all())
            outputs.append(float(values[0, 0]))
        self.assertGreater(outputs[1], 20*outputs[0])

    def test_no_credit_crosses_actual_termination(self):
        zero = jp.zeros((80, 1))
        reward = zero.at[60, 0].set(1.)
        terminal = zero.at[40, 0].set(1.)
        values, _ = compute_gae(zero, terminal, reward, zero, jp.zeros(1),
                                 lambda_=1., discount=.99)
        np.testing.assert_array_equal(values[:41], 0.)
        self.assertGreater(float(values[41, 0]), 0.)


if __name__ == "__main__":
    unittest.main()
