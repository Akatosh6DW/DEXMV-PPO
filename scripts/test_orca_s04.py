import unittest
import numpy as np

from run_orca_s04 import comparisons, jobs


class FocusedBenchmarkTests(unittest.TestCase):
    def test_both_positive_and_negative_results_are_preserved(self):
        matrices = dict(ppo=np.full((10, 100), .3), guided=np.full((10, 100), .8),
                        residual=np.full((10, 100), .2))
        result = comparisons(matrices)
        self.assertTrue(result[0]["supported_on_this_benchmark"])
        self.assertFalse(result[1]["supported_on_this_benchmark"])
        self.assertLess(result[1]["difference_pp"], 0)

    def test_equal_online_budget_and_paired_fresh_starts(self):
        plan = list(jobs({m: dict(std=.3, learning_rate=.0003, shaping=1.) for m in ("ppo", "guided", "residual")}))
        self.assertEqual(len(plan), 41)
        self.assertEqual(len({j["output"] for j in plan}), 41)
        self.assertTrue(all(j["eval_start"] == 81000 and j["eval_count"] == 100 for j in plan))
        for method in ("ppo", "guided", "residual"):
            runs = [j for j in plan if j["method"] == method]
            self.assertEqual(len(runs), 10)
            self.assertTrue(all(j["budget"] == 163840 for j in runs))


if __name__ == "__main__":
    unittest.main()
