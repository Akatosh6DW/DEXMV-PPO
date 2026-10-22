"""Model-selection tests independent of costly simulation."""

import unittest
from pathlib import Path
import tempfile

import numpy as np

from analyze_orca_residual import curve_summary, diagnose


class AnalysisTests(unittest.TestCase):
    def test_first_max_is_selected_and_terminal_is_separate(self):
        rows = [dict(step=s, success30=y) for s, y in ((0, .25), (10, 1), (20, 1), (30, 0))]
        result = curve_summary(rows)
        self.assertEqual(result["best_step"], 10)
        self.assertEqual(result["two_eval_75_step"], 20)
        self.assertEqual(result["last"], 0)
        self.assertAlmostEqual(result["auc"], 2.125/3)

    def test_one_good_eval_not_convergence(self):
        rows = [dict(step=0, success30=1), dict(step=10, success30=0)]
        result = curve_summary(rows)
        self.assertEqual(result["best_step"], 0)
        self.assertIsNone(result["two_eval_75_step"])

    def test_invalid_curve_rejected(self):
        with self.assertRaises(ValueError):
            curve_summary([dict(step=0, success30=0), dict(step=0, success30=1)])

    def test_post_terminal_saturation_not_active_action_saturation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"trace.npz"
            delta = np.array([.01, .015, .02, .08]).reshape(1, 4, 1)
            np.savez(path, alive=[[True, True, False, False]], thumb=np.ones((1, 4)),
                     contacts=np.full((1, 4), 2), forbidden=np.zeros((1, 4)),
                     cube_z=np.ones((1, 4)), spawn_z=np.ones((1, 4)), released=np.ones((1, 4)),
                     requested_residual=delta, effective_residual=delta)
            result = diagnose(path)
            self.assertEqual(result["requested_residual_max_rad"], .08)
            self.assertEqual(result["active_action_requested_max_rad"], .02)
            self.assertEqual(result["active_action_near_bound_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
