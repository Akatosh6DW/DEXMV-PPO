"""Phase exploration must preserve means, late actions, and PPO likelihoods."""

import unittest

import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks
from brax.training.distribution import NormalTanhDistribution

from run_s09_residual_acquisition_noise import acquisition_logits, wrap_network


class AcquisitionNoiseTests(unittest.TestCase):
    def test_early_scale_and_late_logits(self):
        raw = jp.log(jp.expm1(jp.asarray(.029)))
        logits = jp.ones((2, 3, 34)).at[..., 17:].set(raw)
        obs = jp.zeros((2, 3, 123)).at[:, 1, 93].set(.999).at[:, 2, 93].set(1.)
        actual = jax.jit(acquisition_logits, static_argnums=2)(logits, obs, 10.)
        np.testing.assert_array_equal(actual[..., :17], logits[..., :17])
        np.testing.assert_array_equal(actual[:, 2], logits[:, 2])
        dist = NormalTanhDistribution(17).create_dist(actual)
        np.testing.assert_allclose(dist.scale[:, :2], .3, rtol=1e-5)
        np.testing.assert_allclose(dist.scale[:, 2], .03, rtol=1e-5)
        self.assertIs(acquisition_logits(logits, obs, 1.), logits)

    def test_extreme_scales_have_finite_gradients(self):
        obs = jp.zeros((2, 123))
        logits = jp.zeros((2, 34)).at[0, 17:].set(-100.).at[1, 17:].set(100.)
        self.assertTrue(np.isfinite(acquisition_logits(logits, obs, 10.)).all())
        gradient = jax.grad(lambda x: acquisition_logits(x, obs, 10.).sum())(logits)
        self.assertTrue(np.isfinite(gradient).all())

    def test_real_network_uses_identical_sampling_and_loss_distribution(self):
        net = networks.make_ppo_networks(dict(state=123, privileged_state=140), 17,
            policy_obs_key="state", value_obs_key="privileged_state")
        wrapped = wrap_network(net, 10.)
        actor = net.policy_network.init(jax.random.PRNGKey(1))
        obs = dict(state=jp.zeros((2, 123)), privileged_state=jp.zeros((2, 140)))
        normalizer = running_statistics.init_state(dict(state=jp.zeros(123), privileged_state=jp.zeros(140)))
        params, key = (normalizer, actor), jax.random.PRNGKey(2)
        expected = networks.make_inference_fn(net)(params, deterministic=True)(obs, key)[0]
        actual = networks.make_inference_fn(wrapped)(params, deterministic=True)(obs, key)[0]
        np.testing.assert_array_equal(actual, expected)
        _, extras = networks.make_inference_fn(wrapped)(params)(obs, key)
        logits = wrapped.policy_network.apply(*params, obs)
        np.testing.assert_array_equal(extras["distribution_params"], logits)
        np.testing.assert_array_equal(extras["log_prob"],
            wrapped.parametric_action_distribution.log_prob(logits, extras["raw_action"]))
        self.assertIs(wrapped.value_network, net.value_network)

    def test_invalid_configuration(self):
        for multiplier in (0, .5, 11, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                acquisition_logits(jp.zeros(34), jp.zeros(123), multiplier)
        with self.assertRaises(ValueError):
            acquisition_logits(jp.zeros(34), jp.zeros(122), 10.)


if __name__ == "__main__":
    unittest.main()
