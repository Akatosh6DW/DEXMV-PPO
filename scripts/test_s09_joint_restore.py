"""Joint splicing is an explicit diagnostic, with no hidden action averaging."""

import unittest
import jax.numpy as jp
import numpy as np
from probe_s09_joint_restore import splice_actions


class JointRestoreTests(unittest.TestCase):
    def test_selected_outputs_only(self):
        a, b = jp.arange(34).reshape(2, 17), -jp.arange(34).reshape(2, 17)
        mask = jp.arange(17) >= 13
        result = np.asarray(splice_actions(a, b, mask))
        np.testing.assert_array_equal(result[:, :13], np.asarray(a)[:, :13])
        np.testing.assert_array_equal(result[:, 13:], np.asarray(b)[:, 13:])

    def test_reject_incompatible_shapes(self):
        with self.assertRaises(ValueError):
            splice_actions(jp.zeros(16), jp.zeros(16), jp.zeros(17, bool))
        with self.assertRaises(ValueError):
            splice_actions(jp.zeros((2, 17)), jp.zeros(17), jp.zeros(17, bool))


if __name__ == "__main__":
    unittest.main()
