import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
import optax

from run_s09_feedback_adapter_bc import AdapterOptimizer, feedback_only, verify_frozen
from run_s09_recovery_bc import BCOptimizerProxy
from run_s09_mode_experts import BASE_INITIALIZE, expand_parameters
from run_s09_bc_phase_curriculum import KEEP


class FeedbackAdapterTests(unittest.TestCase):
    def test_scope_and_requested_rate(self):
        _, original = BASE_INITIALIZE(SimpleNamespace(arm="residual"), 41, .03)
        params = expand_parameters(original)[1]
        optimizer = BCOptimizerProxy(AdapterOptimizer(optax), 1e-4).adam(1e-3)
        state = optimizer.init(params)
        grads = jax.tree_util.tree_map(jp.ones_like, params)
        changes, state = jax.jit(optimizer.update)(grads, state, params)
        after = optax.apply_updates(params, changes)
        changed = verify_frozen(params, after)
        width = params["experts"][0]["params"]["hidden_0"]["kernel"].shape[1]
        self.assertEqual(changed, 6 * int((~KEEP).sum()) * width)
        for expert in changes["experts"]:
            kernel = np.asarray(expert["params"]["hidden_0"]["kernel"])
            np.testing.assert_array_equal(kernel[KEEP], 0)
            np.testing.assert_allclose(kernel[~KEEP], -1e-4, atol=1e-9, rtol=1e-5)
        wrong = jax.tree_util.tree_map(lambda x: x + .1, after)
        with self.assertRaises(AssertionError):
            verify_frozen(params, wrong)

    def test_projection_is_idempotent_and_rejects_bad_input(self):
        _, original = BASE_INITIALIZE(SimpleNamespace(arm="residual"), 41, .03)
        params = expand_parameters(original)[1]
        once = feedback_only(params)
        twice = feedback_only(once)
        for a, b in zip(jax.tree_util.tree_leaves(once), jax.tree_util.tree_leaves(twice)):
            np.testing.assert_array_equal(a, b)
        malformed = {"experts": [{"params": {"hidden_0": {"kernel": jp.ones((3, 2))}}}]}
        with self.assertRaises(ValueError):
            feedback_only(malformed)


if __name__ == "__main__":
    unittest.main()
