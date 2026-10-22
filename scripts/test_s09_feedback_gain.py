import unittest
from types import SimpleNamespace

import jax
import numpy as np

from make_s09_feedback_gain import KEEP, scaled_experts
from run_s09_mode_experts import BASE_INITIALIZE, expand_parameters


class FeedbackGainTests(unittest.TestCase):
    def test_changes_only_feedback_and_never_source(self):
        _, params = BASE_INITIALIZE(SimpleNamespace(arm="residual"), 41, .03)
        params = expand_parameters(params)
        original = [np.asarray(x).copy() for x in jax.tree_util.tree_leaves(params)]
        scaled, count = scaled_experts(params, .25)
        self.assertGreater(count, 0)
        for a,b in zip(jax.tree_util.tree_leaves(params), original):
            np.testing.assert_array_equal(a,b)
        for a,b in zip(params[1]["experts"], scaled[1]["experts"]):
            x,y = [p["params"]["hidden_0"]["kernel"] for p in (a,b)]
            np.testing.assert_array_equal(np.asarray(x)[KEEP], y[KEEP])
            np.testing.assert_array_equal(np.asarray(x)[~KEEP]*.25, y[~KEEP])
        _, unchanged = scaled_experts(params, 1.)
        self.assertEqual(unchanged, 0)

    def test_bad_gains_rejected(self):
        for gain in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                scaled_experts(None, gain)


if __name__ == "__main__":
    unittest.main()
