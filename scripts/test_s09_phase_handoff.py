"""Handoff boundary, distributions, and frozen-parameter optimizer checks."""

from pathlib import Path
import unittest

from brax.training.agents.ppo import networks
import jax
import jax.numpy as jp
import numpy as np
import optax

import run_s09_phase_handoff as handoff
from run_s09_bc_conditioned import ModeConditionedGrasp


class PhaseHandoffTests(unittest.TestCase):
    def setUp(self):
        model = Path(__file__).resolve().parents[1]/"artifacts/orca_benchmark_suite/S09/inputs/model.xml"
        self.base, self.first = handoff.BASE_INITIALIZE(ModeConditionedGrasp(model, "pure"), 41, .03)
        self.second = jax.tree_util.tree_map(lambda x: jp.array(x), self.first)
        head = self.second[1]["params"]["hidden_2"]
        head["bias"] = head["bias"].at[:17].add(.2)
        self.params = handoff.combine(self.first, self.second)
        self.net = handoff.network_factory(dict(state=(123,), privileged_state=(140,)), 17)
        raw = jax.random.normal(jax.random.PRNGKey(13), (4, 123))
        raw = raw.at[:, 112].set(jp.array([0, 39, 40, 799])/800)
        self.obs = dict(state=raw)

    def test_routing_includes_action_distribution(self):
        logits = self.net.policy_network.apply(*self.params[:2], self.obs)
        a = self.base.policy_network.apply(*self.first[:2], self.obs)
        b = self.base.policy_network.apply(*self.second[:2], self.obs)
        np.testing.assert_array_equal(logits[:2], a[:2])
        np.testing.assert_array_equal(logits[2:], b[2:])
        for deterministic in (True, False):
            for i in range(4):
                obs = dict(state=self.obs["state"][i])
                key = jax.random.PRNGKey(37)
                expected = self.first if i < 2 else self.second
                old = networks.make_inference_fn(self.base)(expected, deterministic)(obs, key)
                new = networks.make_inference_fn(self.net)(self.params, deterministic)(obs, key)
                for x, y in zip(jax.tree_util.tree_leaves(old), jax.tree_util.tree_leaves(new)):
                    np.testing.assert_array_equal(x, y)

    def test_frozen_acquisition_even_after_optimizer_updates(self):
        params = self.params[1]
        optimizer = optax.adam(1e-3)
        state = optimizer.init(params)
        for index in (2, 0, 3):
            obs = dict(state=self.obs["state"][index])
            grads = jax.grad(lambda p: self.net.policy_network.apply(self.params[0], p, obs).sum())(params)
            for x in jax.tree_util.tree_leaves(grads["acquisition"]):
                np.testing.assert_array_equal(x, 0)
            if index == 0:
                for x in jax.tree_util.tree_leaves(grads["feedback"]):
                    np.testing.assert_array_equal(x, 0)
            updates, state = optimizer.update(grads, state, params)
            params = optax.apply_updates(params, updates)
        for a, b in zip(jax.tree_util.tree_leaves(params["acquisition"]),
                        jax.tree_util.tree_leaves(self.first[1])):
            np.testing.assert_array_equal(a, b)
        self.assertTrue(any(np.any(np.asarray(a) != np.asarray(b)) for a, b in zip(
            jax.tree_util.tree_leaves(params["feedback"]), jax.tree_util.tree_leaves(self.second[1]))))

    def test_context_restored_on_failure(self):
        before = handoff.runner.network_factory, handoff.runner.initialize_matched
        with self.assertRaises(RuntimeError):
            with handoff.handoff_networks():
                raise RuntimeError("test")
        self.assertEqual(before, (handoff.runner.network_factory, handoff.runner.initialize_matched))


if __name__ == "__main__":
    unittest.main()
