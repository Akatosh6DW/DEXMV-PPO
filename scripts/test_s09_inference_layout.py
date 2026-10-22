"""Restore preview grouping without changing the numerical tolerance."""

import unittest
import numpy as np
import jax
import jax.numpy as jp
from audit_s09_inference_layout import batch_groups, original_layout_actions


class InferenceLayoutTests(unittest.TestCase):
    def test_noncontiguous_preview_slots_are_restored(self):
        groups = batch_groups([130000, 130001, 130020, 130007], 130000, 64, 8)
        self.assertEqual(groups, [[(0, 0), (1, 1), (3, 7)], [(2, 4)]])
        obs = jp.arange(4*5*123, dtype=jp.float32).reshape(4, 5, 123)
        policy = lambda obs, key: (obs["state"][:17]*.01, {})
        result = jax.jit(lambda x: original_layout_actions(policy, x, groups, 8))(obs)
        np.testing.assert_allclose(result, obs[..., :17]*.01, atol=0, rtol=0)

    def test_invalid_seeds_rejected(self):
        for seeds in ([130000, 130000], [129999], [130064]):
            with self.assertRaises(ValueError):
                batch_groups(seeds, 130000, 64, 8)


if __name__ == "__main__":
    unittest.main()
