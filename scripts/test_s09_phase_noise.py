"""The exploration schedule must not alter means or likelihood semantics."""

import unittest

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import networks
from brax.training.distribution import NormalTanhDistribution
from brax.training.acme import running_statistics

from run_s09_phase_noise import phase_logits, wrap_network


class PhaseNoiseTests(unittest.TestCase):
    def test_mean_late_logits_and_early_std(self):
        raw = jp.log(jp.expm1(jp.asarray(.029)))
        logits = jp.ones((2, 3, 34)).at[..., 17:].set(raw)
        obs = jp.zeros((2, 3, 123)).at[:, 1, 93].set(.999).at[:, 2, 93].set(1.)
        changed = jax.jit(phase_logits, static_argnums=2)(logits, obs, .1)
        np.testing.assert_array_equal(changed[..., :17], logits[..., :17])
        np.testing.assert_array_equal(changed[:, 2], logits[:, 2])
        dist = NormalTanhDistribution(17).create_dist(changed)
        np.testing.assert_allclose(dist.scale[:, :2], .003, rtol=1e-5)
        np.testing.assert_allclose(dist.scale[:, 2], .03, rtol=1e-5)
        tiny = phase_logits(logits.at[..., 17:].set(-100.), obs, .1)
        self.assertTrue(np.isfinite(tiny).all())
        gradient = jax.grad(lambda x: phase_logits(x, obs, .1).sum())(logits)
        self.assertTrue(np.isfinite(gradient).all())
        self.assertTrue(np.all(np.asarray(gradient[..., 17:]) > 0))

    def test_actual_network_sampling_and_deterministic_actions(self):
        network = networks.make_ppo_networks(dict(state=123, privileged_state=140), 17,
            policy_obs_key="state", value_obs_key="privileged_state")
        wrapped = wrap_network(network, .1)
        actor = network.policy_network.init(jax.random.PRNGKey(1))
        obs = dict(state=jp.zeros((2, 123)), privileged_state=jp.zeros((2, 140)))
        normalizer = running_statistics.init_state(dict(state=jp.zeros(123), privileged_state=jp.zeros(140)))
        params = (normalizer, actor)
        key = jax.random.PRNGKey(2)
        baseline = networks.make_inference_fn(network)(params, deterministic=True)(obs, key)[0]
        actual = networks.make_inference_fn(wrapped)(params, deterministic=True)(obs, key)[0]
        np.testing.assert_array_equal(actual, baseline)
        _, extras = networks.make_inference_fn(wrapped)(params)(obs, key)
        logits = wrapped.policy_network.apply(*params, obs)
        np.testing.assert_array_equal(extras["distribution_params"], logits)
        np.testing.assert_array_equal(extras["log_prob"],
            wrapped.parametric_action_distribution.log_prob(logits, extras["raw_action"]))
        self.assertIs(wrapped.value_network, network.value_network)

    def test_invalid_settings(self):
        for scale in (0, -1, 2, float("nan")):
            with self.assertRaises(ValueError):
                phase_logits(jp.zeros(34), jp.zeros(123), scale)
        with self.assertRaises(ValueError):
            phase_logits(jp.zeros(34), jp.zeros(122), .1)


if __name__ == "__main__":
    unittest.main()
