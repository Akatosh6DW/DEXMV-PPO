"""Check cloned behavior, routing, gradients and distribution consistency."""

import unittest
from types import SimpleNamespace

import run_s09_mode_experts as experts
from make_s09_mode_experts import verify_copies
from brax.training.agents.ppo import networks
import jax
import jax.numpy as jp
import numpy as np


class ModeExpertTests(unittest.TestCase):
    def setUp(self):
        self.base, self.params = experts.BASE_INITIALIZE(SimpleNamespace(arm="residual"), 41, .03)
        head = self.params[1]["params"]["hidden_2"]
        head["kernel"] = head["kernel"].at[:, :17].set(
            .01 * jax.random.normal(jax.random.PRNGKey(25), (128, 17)))
        self.net = experts.network_factory(dict(state=(123,), privileged_state=(140,)), 17)
        self.expanded = experts.expand_parameters(self.params)
        raw = jax.random.normal(jax.random.PRNGKey(17), (6, 123))
        raw = raw.at[:, 95:101].set(jp.eye(6))
        self.obs = dict(state=raw, privileged_state=jp.pad(raw, ((0, 0), (0, 17))))

    def test_copies_critic_and_normalizer_unchanged(self):
        verify_copies(self.params, self.expanded)
        with self.assertRaises(ValueError):
            experts.expand_parameters(self.expanded)

    def test_all_modes_preserve_actions_scales_and_log_prob(self):
        for deterministic in (True, False):
            key = jax.random.PRNGKey(99)
            old = networks.make_inference_fn(self.base)(self.params, deterministic)(self.obs, key)
            new = networks.make_inference_fn(self.net)(self.expanded, deterministic)(self.obs, key)
            for x, y in zip(jax.tree_util.tree_leaves(old), jax.tree_util.tree_leaves(new)):
                np.testing.assert_array_equal(x, y)
        for i in range(6):
            obs = jax.tree_util.tree_map(lambda x:x[i], self.obs)
            a = self.net.policy_network.apply(*self.expanded[:2], obs)
            b = self.base.policy_network.apply(*self.params[:2], obs)
            np.testing.assert_array_equal(a, b)

    def test_selected_expert_only_controls_outputs_and_gradients(self):
        before = self.net.policy_network.apply(*self.expanded[:2], self.obs)
        copies = list(self.expanded[1]["experts"])
        copies[2] = jax.tree_util.tree_map(lambda x:x.copy(), copies[2])
        head = copies[2]["params"]["hidden_2"]
        head["bias"] = head["bias"].at[:17].add(.2)
        changed = {"experts": tuple(copies)}
        after = self.net.policy_network.apply(self.params[0], changed, self.obs)
        np.testing.assert_array_equal(before[[0, 1, 3, 4, 5], :], after[[0, 1, 3, 4, 5], :])
        self.assertGreater(float(jp.max(jp.abs(after[2]-before[2]))), .1)
        obs = jax.tree_util.tree_map(lambda x:x[2:3], self.obs)
        grads = jax.grad(lambda p: self.net.policy_network.apply(self.params[0], p, obs).sum())(changed)
        for i, grad in enumerate(grads["experts"]):
            norm = sum(float(jp.sum(jp.abs(x))) for x in jax.tree_util.tree_leaves(grad))
            self.assertGreater(norm, 0) if i == 2 else self.assertEqual(norm, 0)

    def test_patch_restores_functions(self):
        factory, init = experts.runner.network_factory, experts.runner.initialize_matched
        with self.assertRaises(RuntimeError):
            with experts.expert_networks():
                self.assertIs(experts.runner.network_factory, experts.network_factory)
                raise RuntimeError("restore even on error")
        self.assertIs(experts.runner.network_factory, factory)
        self.assertIs(experts.runner.initialize_matched, init)

    def test_feedback_is_not_removed_by_routing(self):
        other = dict(self.obs, state=self.obs["state"].at[:, :17].add(.5))
        before = self.net.policy_network.apply(*self.expanded[:2], self.obs)
        after = self.net.policy_network.apply(*self.expanded[:2], other)
        self.assertGreater(float(jp.max(jp.abs(after-before))), 1e-6)


if __name__ == "__main__":
    unittest.main()
