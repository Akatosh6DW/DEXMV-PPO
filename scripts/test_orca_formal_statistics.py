"""Guard against pseudoreplication and overly small exact p-values."""

import unittest
import numpy as np

from orca_formal_statistics import paired_effect, seed_estimate


class SeedStatisticsTests(unittest.TestCase):
    def test_seed_mean_standard_deviation_and_interval(self):
        result = seed_estimate([0, .25, .5, .75, 1])
        self.assertEqual(result["n_training_seeds"], 5)
        self.assertEqual(result["mean"], .5)
        self.assertAlmostEqual(result["std"], np.std([0, .25, .5, .75, 1], ddof=1))
        self.assertLess(result["bootstrap95"][0], .5)
        self.assertGreater(result["bootstrap95"][1], .5)
        with self.assertRaises(ValueError):
            seed_estimate(np.zeros((5, 64)))

    def test_exact_minimum_p_and_pairing(self):
        result = paired_effect(dict.fromkeys(range(5), 1.), dict.fromkeys(range(5), 0.))
        self.assertEqual(result["exact_two_sided_signflip_p"], .0625)
        self.assertEqual(result["minimum_two_sided_p"], .0625)
        same = paired_effect(dict.fromkeys(range(5), 1.), dict.fromkeys(range(5), 1.))
        self.assertEqual(same["exact_two_sided_signflip_p"], 1.)
        with self.assertRaises(ValueError):
            paired_effect({1: 1}, {2: 1})


if __name__ == "__main__":
    unittest.main()
