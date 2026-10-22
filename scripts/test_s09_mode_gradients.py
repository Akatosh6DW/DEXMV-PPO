"""The diagnostic decomposition must match the installed PPO actor gradient."""

from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jp
import numpy as np
from brax.training.types import Transition
from brax.training.agents.ppo import networks
from diagnose_s09_mode_gradients import gradient_measurement, summarize_gradients
from train_orca_matched import initialize_matched


class ModeGradientTests(unittest.TestCase):
    def test_matches_original_actor_gradient(self):
        net, params = initialize_matched(SimpleNamespace(arm="residual"), 51, .03)
        raw = jax.random.normal(jax.random.PRNGKey(12), (6, 4, 123))
        raw = raw.at[..., 95:101].set(jp.broadcast_to(jp.eye(6)[:, None, :], (6, 4, 6)))
        obs = dict(state=raw, privileged_state=jp.pad(raw, ((0, 0), (0, 0), (0, 17))))
        action, extras = networks.make_inference_fn(net)(params)(obs, jax.random.PRNGKey(8))
        data = Transition(observation=obs, action=action, reward=jp.arange(24).reshape(6, 4)*.01,
            discount=jp.ones((6, 4)), next_observation=obs,
            extras=dict(policy_extras=extras, state_extras=dict(truncation=jp.zeros((6, 4)))))
        matrix, standard, _, modes = gradient_measurement(net, params, data, jax.random.PRNGKey(15))
        np.testing.assert_allclose(np.asarray(matrix.sum(axis=0)), standard, atol=2e-5, rtol=2e-4)
        np.testing.assert_array_equal(np.bincount(np.asarray(modes).ravel(), minlength=6), [4]*6)

    def test_cosine_and_first_order_sign(self):
        matrix = np.zeros((6, 2))
        matrix[0, 0], matrix[1, 0] = 1, -2
        report = summarize_gradients(matrix, [1, 1, 0, 0, 0, 0])
        self.assertEqual(report["cosine"][0][1], -1)
        self.assertGreater(report["full_batch_sgd_first_order_loss_change"][0], 0)
        self.assertIsNone(report["cosine"][0][2])

    def test_reject_bad_arrays(self):
        with self.assertRaises(ValueError):
            summarize_gradients(np.zeros((5, 2)), [1]*6)
        with self.assertRaises(ValueError):
            summarize_gradients(np.full((6, 2), np.nan), [1]*6)


if __name__ == "__main__":
    unittest.main()
