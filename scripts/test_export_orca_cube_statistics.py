"""Focused tests for report exports, separate from frozen training/evaluation code."""

import unittest

from export_orca_cube_statistics import metric_rows, md_table, validate, SEEDS


class CubeExportTests(unittest.TestCase):
    def test_seed_order_and_units_are_preserved(self):
        values = [0.0, 0.25, 0.5, 0.75, 1.0]
        data = dict(table=[dict(condition="ppo_on", scenario="nominal",
                               invalid_initial_counts={str(s): i for i, s in enumerate(SEEDS)},
                               metrics=dict(success30=dict(individual=values, mean=.5, std=.2,
                                                           bootstrap95=[.2, .8], n_training_seeds=5)))])
        aggregate, individual = metric_rows(data)
        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate[0]["mean"], .5)
        self.assertEqual(aggregate[0]["ci95_low"], .2)
        self.assertEqual([r["train_seed"] for r in individual], list(SEEDS))
        self.assertEqual([r["value"] for r in individual], values)
        self.assertEqual([r["invalid_initial_count"] for r in individual], list(range(5)))

    def test_rejects_changed_statistics(self):
        with self.assertRaisesRegex(ValueError, "differ"):
            validate({"evaluations": 512}, {"evaluations": 513}, {})

    def test_rejects_partial_statistics(self):
        data = dict(evaluations_complete=False, missing=["missing"], evaluations=512)
        with self.assertRaisesRegex(ValueError, "513"):
            validate(data, data, {})

    def test_table_keeps_zero_not_missing(self):
        table = md_table(["seed", "success"], [[301, 0], [302, 1]])
        self.assertEqual(table[2], "| 301 | 0 |")
        self.assertEqual(len(table), 4)


if __name__ == "__main__":
    unittest.main()
