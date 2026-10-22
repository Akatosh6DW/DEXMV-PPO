"""Sampling, event dimensions, KL direction, and unchanged deployment means."""

import unittest

import jax
import jax.numpy as jp
import numpy as np
from scipy.stats import multivariate_normal
from brax.training.distribution import NormalTanhDistribution
from brax.training.agents.ppo import networks
from brax.training.acme import running_statistics

from run_s09_correlated_exploration import CorrelatedTanh, head_correlation, wrap_network


class CorrelatedExplorationTests(unittest.TestCase):
    def setUp(self):
        self.matrix = head_correlation(np.random.default_rng(42).normal(size=(128,34)), .5)
        self.logits = jp.concatenate([jp.linspace(-.5,.5,17), jp.full(17,np.log(np.expm1(.029)))])
        self.key = jax.random.PRNGKey(3)

    def test_correlation_and_invalid_inputs(self):
        np.testing.assert_allclose(np.diag(self.matrix), 1.)
        self.assertGreaterEqual(np.linalg.eigvalsh(self.matrix).min(), .5)
        for strength in (-1, 1, float("nan")):
            with self.assertRaises(ValueError):
                head_correlation(np.ones((128,34)), strength)
        np.testing.assert_array_equal(head_correlation(np.zeros((128,34)), .5),np.eye(17))
        zero_column = np.ones((128,34))
        zero_column[:,0] = 0
        np.testing.assert_array_equal(head_correlation(zero_column,.5)[0],np.eye(17)[0])
        with self.assertRaises(ValueError):
            CorrelatedTanh(np.ones((17,17)))

    def test_identity_equivalence(self):
        original, wrapped = NormalTanhDistribution(17), CorrelatedTanh(np.eye(17))
        params = jp.broadcast_to(self.logits,(2,3,34))
        raw = original.sample_no_postprocessing(params, self.key)
        np.testing.assert_array_equal(wrapped.mode(params), original.mode(params))
        np.testing.assert_allclose(wrapped.sample_no_postprocessing(params,self.key), raw, atol=1e-7)
        np.testing.assert_allclose(wrapped.log_prob(params,raw), original.log_prob(params,raw), atol=5e-5)
        np.testing.assert_allclose(wrapped.entropy(params,self.key), original.entropy(params,self.key), atol=1e-5)
        self.assertEqual(wrapped.log_prob(params,raw).shape, (2,3))

    def test_density_kl_and_empirical_covariance(self):
        dist = CorrelatedTanh(self.matrix)
        params = jp.broadcast_to(self.logits,(32768,34))
        samples = np.asarray(jax.jit(dist.sample_no_postprocessing)(params,self.key))
        np.testing.assert_allclose(np.cov(samples.T)/(.03**2), self.matrix, atol=.026)
        raw = samples[0]
        expected = multivariate_normal.logpdf(raw, np.asarray(self.logits[:17]), .03**2*self.matrix)
        jac = np.asarray(dist._postprocessor.forward_log_det_jacobian(jp.asarray(raw))).sum()
        self.assertAlmostEqual(float(dist.log_prob(self.logits,jp.asarray(raw))), expected-jac, places=4)
        old = dist.create_dist(self.logits)
        new_logits = self.logits.at[:17].add(.01)
        new = dist.create_dist(new_logits)
        expected_kl = .5*np.ones(17).dot(np.linalg.solve(.03**2*self.matrix,np.ones(17)))*.01**2
        self.assertAlmostEqual(float(new.kl_divergence(old)), expected_kl, places=4)
        self.assertAlmostEqual(float(old.kl_divergence(old)), 0., places=5)
        varied = dist.create_dist(new_logits.at[17:].add(jp.linspace(-.2,.3,17)))
        a, b = np.asarray(old.cov, dtype=float), np.asarray(varied.cov, dtype=float)
        diff = np.asarray(varied.loc-old.loc, dtype=float)
        expected_kl = .5*(np.trace(np.linalg.solve(b,a)) + diff.dot(np.linalg.solve(b,diff)) -
                           17 + np.linalg.slogdet(b)[1]-np.linalg.slogdet(a)[1])
        self.assertAlmostEqual(float(varied.kl_divergence(old)), expected_kl, places=4)
        self.assertGreater(abs(float(varied.kl_divergence(old)-old.kl_divergence(varied))), .01)
        gradient = jax.grad(lambda x: dist.log_prob(x,jp.asarray(raw))+.01*dist.entropy(x,self.key))(self.logits)
        self.assertTrue(np.isfinite(gradient).all())

    def test_network_sampling_and_means(self):
        network = networks.make_ppo_networks(dict(state=123,privileged_state=140),17,
            policy_obs_key="state",value_obs_key="privileged_state")
        wrapped = wrap_network(network,self.matrix)
        normalizer = running_statistics.init_state(dict(state=jp.zeros(123),privileged_state=jp.zeros(140)))
        params = normalizer, network.policy_network.init(self.key)
        obs = dict(state=jp.ones((2,123)),privileged_state=jp.ones((2,140)))
        standard = networks.make_inference_fn(network)(params,deterministic=True)(obs,self.key)[0]
        actual = networks.make_inference_fn(wrapped)(params,deterministic=True)(obs,self.key)[0]
        np.testing.assert_array_equal(actual,standard)
        _,extras = networks.make_inference_fn(wrapped)(params)(obs,self.key)
        logits = wrapped.policy_network.apply(*params,obs)
        np.testing.assert_array_equal(extras["log_prob"],wrapped.parametric_action_distribution.log_prob(logits,extras["raw_action"]))
        kl = jax.jit(lambda x: wrapped.parametric_action_distribution.create_dist(x).kl_divergence(
            wrapped.parametric_action_distribution.create_dist(logits)))(logits.at[..., :17].add(.01))
        self.assertEqual(kl.shape,(2,))
        self.assertTrue(np.isfinite(kl).all())
        self.assertIs(network.policy_network,wrapped.policy_network)
        self.assertIs(network.value_network,wrapped.value_network)


if __name__ == "__main__":
    unittest.main()
