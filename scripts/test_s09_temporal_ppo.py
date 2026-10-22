"""Temporal partitioning preserves physical and gradient batch budgets."""

import unittest

from run_s09_temporal_ppo import temporal_options


class TemporalTests(unittest.TestCase):
    def test_same_transition_counts_and_valid_env_partition(self):
        for length in (20, 40, 80):
            options = temporal_options(length)
            self.assertEqual(options["batch_size"] * options["unroll_length"], 640)
            self.assertEqual(options["batch_size"] * length * 4, 2560)
            self.assertEqual((options["batch_size"] * 4) % 32, 0)
        self.assertEqual(temporal_options(20), dict(unroll_length=20, batch_size=32))

    def test_invalid_partitions_rejected(self):
        for length in (0, 10, 100, 800):
            with self.assertRaises(ValueError):
                temporal_options(length)


if __name__ == "__main__":
    unittest.main()
