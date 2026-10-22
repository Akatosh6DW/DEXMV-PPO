"""Test paired-case inference without pretending old/new model seeds are paired."""

import unittest

import numpy as np

from analyze_s09_bc_confirmation import bootstrap


class ConfirmationTests(unittest.TestCase):
    def test_identical_paired_models_have_zero_difference(self):
        x = np.asarray([[0,1,0,1], [1,1,0,0], [0,0,1,0]], float)
        np.testing.assert_array_equal(bootstrap(x, x, paired_models=True, draws=100), [0,0])

    def test_independent_training_seeds_are_not_forced_equal(self):
        x = np.asarray([[0,0,0,0], [1,1,1,1], [0,0,0,0]], float)
        lo, hi = bootstrap(x, x, paired_models=False, draws=400)
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)

    def test_different_numbers_of_models_share_test_states(self):
        x = np.ones((3,8))
        y = np.zeros((1,8))
        np.testing.assert_array_equal(bootstrap(x, y, draws=100), [1,1])


if __name__ == "__main__":
    unittest.main()
