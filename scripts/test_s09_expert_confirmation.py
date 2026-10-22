import unittest

import numpy as np

from analyze_s09_expert_confirmation import holm


class ExpertConfirmationTests(unittest.TestCase):
    def test_holm_preserves_original_order(self):
        np.testing.assert_allclose(holm([.04, .001, .03]), [.06, .003, .06])
        np.testing.assert_array_equal(holm([1., 1., 1.]), 1.)

    def test_invalid_probabilities(self):
        for values in ([], [-.1], [1.01], [float("nan")]):
            with self.assertRaises(ValueError):
                holm(values)


if __name__ == "__main__":
    unittest.main()
