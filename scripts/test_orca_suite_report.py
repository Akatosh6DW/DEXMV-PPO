"""CPU-only tests for the frozen-comparison statistical helpers."""

import unittest

import numpy as np

from report_orca_suite import bootstrap_interval, holm, sign_flip_pvalue


class StatisticsTests(unittest.TestCase):
    def test_constant_difference(self):
        np.testing.assert_allclose(bootstrap_interval(np.ones((10, 200))), [1., 1.])
        self.assertEqual(bootstrap_interval(np.zeros((1, 200))), [0., 0.])

    def test_paired_zero_control_and_determinism(self):
        rng = np.random.default_rng(45)
        a = rng.integers(0, 2, (10, 200))
        b = rng.integers(0, 2, (1, 200))
        low, high = bootstrap_interval(a-b)
        self.assertLess(low, np.mean(a-b))
        self.assertGreater(high, np.mean(a-b))
        self.assertEqual([low, high], bootstrap_interval(a-b))

    def test_exact_test_resolution(self):
        self.assertEqual(sign_flip_pvalue(np.ones(10)), 2/1024)
        self.assertEqual(sign_flip_pvalue(np.zeros(10)), 1.)
        self.assertEqual(sign_flip_pvalue([1, -1]), 1.)
        self.assertEqual(sign_flip_pvalue([-1]*10), 2/1024)

    def test_holm_order_and_known_values(self):
        np.testing.assert_allclose(holm([.04, .01, .03]), [.06, .03, .06])
        self.assertGreater(holm([2/1024]*30)[0], .05)
        self.assertLess(holm([2/1024]*5)[0], .05)

    def test_invalid_inputs(self):
        for values in ([], [np.nan], [2]):
            if values:
                with self.assertRaises(ValueError):
                    holm(values)
        for values in ([], [np.nan], np.ones((2, 2))):
            with self.assertRaises(ValueError):
                sign_flip_pvalue(values)
        with self.assertRaises(ValueError):
            bootstrap_interval(np.empty((0, 10)))


if __name__ == "__main__":
    unittest.main()
