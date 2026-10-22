"""The balancing changes probabilities, never the reference physical states."""

import unittest

import numpy as np

from run_s09_balanced_reference_starts import balanced_indices


class BalancedReferenceStartsTests(unittest.TestCase):
    def test_equal_classes_and_equal_within_class(self):
        modes = np.array([0, 0, 0, 1, 2, 2])
        indices, counts = balanced_indices(modes)
        np.testing.assert_array_equal(counts, [3, 1, 2, 0, 0, 0])
        np.testing.assert_array_equal(np.bincount(modes[indices], minlength=6), [6, 6, 6, 0, 0, 0])
        np.testing.assert_array_equal(np.bincount(indices), [2, 2, 2, 6, 3, 3])

    def test_real_bank_counts_no_fabricated_missing_class(self):
        modes = np.repeat(np.arange(6), [40, 30, 1, 6, 0, 6])
        indices, _ = balanced_indices(modes)
        self.assertEqual(len(indices), 600)
        np.testing.assert_array_equal(np.bincount(modes[indices], minlength=6), [120, 120, 120, 120, 0, 120])
        np.testing.assert_array_equal(np.unique(indices), np.arange(83))

    def test_invalid_and_excessive_banks(self):
        for modes in ([], [-1], [6], [.5], [[0]], np.repeat([0, 1, 2], [23, 29, 31])):
            with self.assertRaises(ValueError):
                balanced_indices(modes)


if __name__ == "__main__":
    unittest.main()
