"""Mode anchoring must leave unselected examples and inference unchanged."""

from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jp
import numpy as np

from run_s09_bc_trust import trusted_loss, with_std
from run_s09_mode_anchor import mode_trusted_loss
from train_orca_matched import initialize_matched


class ModeAnchorTests(unittest.TestCase):
    def setUp(self):
        self.net, self.teacher = initialize_matched(SimpleNamespace(arm="residual"), 41, .03)
        self.changed = with_std(self.teacher, .06)
        self.original = lambda *a, **k: (jp.float32(2), {"original": jp.float32(2)})

    def evaluate(self, modes, selected, changed=True):
        obs = jp.ones((len(modes), 123)).at[:, 95:101].set(jax.nn.one_hot(jp.asarray(modes), 6))
        data = SimpleNamespace(observation={"state": obs})
        params = self.changed if changed else self.teacher
        fn = mode_trusted_loss(self.original, self.teacher, .1, selected)
        return fn(SimpleNamespace(policy=params[1]), params[0], data,
                  jax.random.PRNGKey(0), ppo_network=self.net), data

    def test_unselected_and_initial_penalty_zero(self):
        for modes, changed in [([1, 2, 3], True), ([0, 0], False)]:
            (value, metrics), _ = self.evaluate(modes, [0], changed)
            self.assertAlmostEqual(float(value), 2., places=6)
            self.assertAlmostEqual(float(metrics["bc_teacher_penalty"]), 0., places=6)

    def test_full_gate_matches_existing_teacher_loss(self):
        (value, metrics), data = self.evaluate([0, 1, 2, 3, 4, 5], list(range(6)))
        old, _ = trusted_loss(self.original, self.teacher, .1)(
            SimpleNamespace(policy=self.changed[1]), self.changed[0], data,
            jax.random.PRNGKey(0), ppo_network=self.net)
        np.testing.assert_array_equal(value, old)
        self.assertEqual(float(metrics["anchor_fraction"]), 1.)

    def test_selected_samples_only_contribute_to_penalty(self):
        (_, metrics), _ = self.evaluate([0, 0], [0])
        (_, mixed), _ = self.evaluate([0, 1], [0])
        self.assertAlmostEqual(float(mixed["anchor_fraction"]), .5)
        self.assertAlmostEqual(float(mixed["bc_teacher_penalty"]),
                               float(metrics["bc_teacher_penalty"])/2, places=5)

    def test_invalid_configuration_rejected(self):
        for weight, modes in [(0, [0]), (float("nan"), [0]), (.1, []), (.1, [6]), (.1, [0, 0])]:
            with self.assertRaises(ValueError):
                mode_trusted_loss(self.original, self.teacher, weight, modes)


if __name__ == "__main__":
    unittest.main()
