"""Convergence is sustained success, not a single lucky checkpoint."""

import csv
from pathlib import Path
import tempfile
import unittest

from analyze_orca_comparison import summarize


class CurveTests(unittest.TestCase):
    def summary(self, successes):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            with (path / "metrics.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=["step", "success30"])
                writer.writeheader()
                writer.writerows(dict(step=i * 100, success30=s)
                                 for i, s in enumerate(successes))
            return summarize(path)

    def test_isolated_success_is_censored(self):
        result = self.summary([0, 1, 0])
        self.assertTrue(result["censored"])
        self.assertIsNone(result["convergence_onset_step"])
        self.assertEqual(result["best_step"], 100)

    def test_sustained_success_has_separate_confirmation(self):
        result = self.summary([0, .8125, .9375, 0])
        self.assertEqual(result["convergence_onset_step"], 100)
        self.assertEqual(result["convergence_confirmed_step"], 200)
        self.assertEqual(result["last"], 0)

    def test_area_and_step_zero(self):
        self.assertEqual(self.summary([0, 1, 1])["normalized_auc"], .75)
        result = self.summary([1, 1, 1])
        self.assertEqual(result["convergence_onset_step"], 0)
        self.assertEqual(result["normalized_auc"], 1)


if __name__ == "__main__":
    unittest.main()
