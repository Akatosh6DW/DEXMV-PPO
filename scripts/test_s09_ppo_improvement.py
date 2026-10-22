"""A discount ablation must not silently alter other PPO settings."""

import unittest

from run_s09_ppo_improvement import training_options


class ImprovementTests(unittest.TestCase):
    def test_only_discount_changes(self):
        self.assertEqual(training_options(.99), {"discounting":.99})
        self.assertEqual(training_options(.999), {"discounting":.999})

    def test_rejects_invalid_discount(self):
        for value in (0, -1, 1, 2, float("nan")):
            with self.assertRaises(ValueError):
                training_options(value)


if __name__ == "__main__":
    unittest.main()
