import unittest
from types import SimpleNamespace

import jax
import numpy as np
import optax

import run_s09_mode_experts as experts
from run_s09_recovery_expert_bc import fit_expanded


class RecoveryExpertTests(unittest.TestCase):
    def test_optimizer_forwarded_and_restored(self):
        _, params = experts.BASE_INITIALIZE(SimpleNamespace(arm="residual"), 41, .03)
        proxy = object()
        received = []

        def original(net, expanded, arrays, updates, seed):
            received.append(optax)
            self.assertEqual(len(expanded[1]["experts"]), 6)
            self.assertEqual((updates, seed), (20, 13112))
            for expert in expanded[1]["experts"]:
                for a,b in zip(jax.tree_util.tree_leaves(expert), jax.tree_util.tree_leaves(params[1])):
                    np.testing.assert_array_equal(a, b)
            return expanded, []

        module = original.__globals__["optax"]
        fit_expanded(original, None, params, (), 20, 13112, proxy)
        self.assertIs(received[0], proxy)
        self.assertIs(original.__globals__["optax"], module)

    def test_restored_after_fit_failure(self):
        _, params = experts.BASE_INITIALIZE(SimpleNamespace(arm="residual"), 41, .03)

        def original(*args):
            raise RuntimeError("fit failed")

        module = original.__globals__["optax"]
        with self.assertRaises(RuntimeError):
            fit_expanded(original, None, params, (), 20, 13112, object())
        self.assertIs(original.__globals__["optax"], module)


if __name__ == "__main__":
    unittest.main()
