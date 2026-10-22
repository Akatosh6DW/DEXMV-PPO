"""Calibration aggregation excludes padding and preserves signed bias."""

import unittest
import numpy as np

from diagnose_s09_learned_critic import calibration


class CalibrationTests(unittest.TestCase):
    def test_mask_bias_and_rmse(self):
        result = calibration([[3., 999.], [1., 0.]], [[1., -999.], [3., 0.]],
                             [[True, False], [True, False]])
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["rmse"], 2.)
        self.assertEqual(result["bias"], 0.)
        self.assertEqual(result["mean_prediction"], 2.)
        self.assertEqual(calibration([0], [0], [False])["n"], 0)

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            calibration([1], [[1]], [True])
        with self.assertRaises(ValueError):
            calibration([np.nan], [1], [True])


if __name__ == "__main__":
    unittest.main()
